"""
main.py
--------
Entry point — ket noi tat ca thanh phan va chay training loop.

Usage:
    python main.py                                # GA mode (default)
    python main.py --config configs/config.yaml
    python main.py --algorithm ppo --headless     # PPO mode
    python main.py --track figure8 --headless --generations 50
    python main.py --resume                       # Tiep tuc tu checkpoint
"""

import argparse
import yaml
import sys
import numpy as np
from pathlib import Path

# -- Them src vao path -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from src.core.neural_network import NeuralNetwork
from src.core.genetic_algorithm import GeneticAlgorithm
from src.simulation.car import Car
from src.simulation.track import Track
from mlflow_tracking import MLflowTracker

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_config(cfg: dict) -> None:
    """Ghi config xuong class-level constants."""
    sim = cfg.get("simulation", {})
    Car.MAX_SPEED       = sim.get("max_speed", Car.MAX_SPEED)
    Car.ACCELERATION    = sim.get("acceleration", Car.ACCELERATION)
    Car.FRICTION        = sim.get("friction", Car.FRICTION)
    Car.MAX_STEER       = sim.get("max_steer", Car.MAX_STEER)
    Car.N_SENSORS       = sim.get("n_sensors", Car.N_SENSORS)
    Car.SENSOR_RANGE    = sim.get("sensor_range", Car.SENSOR_RANGE)
    Car.SENSOR_ANGLES   = sim.get("sensor_angles", Car.SENSOR_ANGLES)
    Car.MAX_STUCK_FRAMES = sim.get("max_stuck_frames", Car.MAX_STUCK_FRAMES)


# ============================================================================
#  GA MODE
# ============================================================================

def make_population(ga: GeneticAlgorithm, track: Track) -> list[Car]:
    """Tao list Car tu population NN cua GA."""
    cars = []
    for i, nn in enumerate(ga.population):
        car = Car(
            nn=nn,
            start_x=track.start_x,
            start_y=track.start_y,
            start_angle=track.start_angle,
            is_elite=(i < ga.n_elites and ga.generation > 0),
        )
        cars.append(car)
    return cars


def run_headless(ga: GeneticAlgorithm, track: Track, max_frames: int = 3000) -> tuple[list[float], int]:
    """Chay 1 generation khong render, tra ve (fitness scores, alive count)."""
    cars = make_population(ga, track)
    for _ in range(max_frames):
        all_dead = all(not c.alive for c in cars)
        if all_dead:
            break
        for car in cars:
            car.step(track)
    alive_count = sum(1 for c in cars if c.alive)
    return [c.fitness for c in cars], alive_count


def run_with_render(
    ga: GeneticAlgorithm,
    track: Track,
    renderer,
    hud,
    controls,
    max_frames: int = 3000,
) -> tuple[list[float], int] | None:
    """Chay 1 generation voi pygame rendering, tra ve (fitness scores, alive count)."""
    import pygame
    cars = make_population(ga, track)

    for frame in range(max_frames):
        events = pygame.event.get()
        if controls.handle_events(events):
            return None  # signal to stop (quit)

        if controls.paused:
            # Ve lai frame hien tai khi pause
            renderer.draw_frame(cars, track, {})
            hud.draw(renderer.screen, {}, len([c for c in cars if c.alive]), len(cars))
            controls.draw(renderer.screen)
            pygame.display.flip()
            renderer.clock.tick(renderer.fps)
            continue

        for _ in range(controls.speed):
            all_dead = all(not c.alive for c in cars)
            if all_dead:
                break
            for car in cars:
                car.step(track)

        if all(not c.alive for c in cars):
            break

        alive = [c for c in cars if c.alive]
        current_best = max(c.fitness for c in cars)
        stats = {
            "generation": ga.generation + 1,
            "current_best": current_best,
            "all_time_best": max(ga.best_fitness_history + [current_best]) if ga.best_fitness_history else current_best,
        }
        renderer.draw_frame(cars, track, stats)
        hud.draw(renderer.screen, stats, len(alive), len(cars))
        controls.draw(renderer.screen)
        pygame.display.flip()

    alive_count = sum(1 for c in cars if c.alive)
    return [c.fitness for c in cars], alive_count


def run_ga(args, cfg):
    """Training loop cho Genetic Algorithm mode."""
    ga_cfg  = cfg.get("genetic_algorithm", {})
    nn_cfg  = cfg.get("neural_network", {})
    trk_cfg = cfg.get("track", {})
    trn_cfg = cfg.get("training", {})
    ren_cfg = cfg.get("rendering", {})

    headless   = args.headless or trn_cfg.get("headless", False)
    max_gens   = args.generations or trn_cfg.get("max_generations", 200)
    track_name = args.track or trk_cfg.get("default", "oval")
    speed      = args.speed
    save_every = trn_cfg.get("save_best_every", 10)
    output_dir = trn_cfg.get("output_dir", "./checkpoints")

    # -- Init --
    track = Track.from_preset(track_name, trk_cfg.get("track_width", 56))
    track.center(ren_cfg.get("width", 1280), ren_cfg.get("height", 720))

    ga = GeneticAlgorithm(
        population_size  = ga_cfg.get("population_size", 20),
        n_elites         = ga_cfg.get("n_elites", 4),
        mutation_rate    = ga_cfg.get("mutation_rate", 0.08),
        mutation_strength= ga_cfg.get("mutation_strength", 0.30),
        crossover_prob   = ga_cfg.get("crossover_prob", 0.50),
        nn_architecture  = nn_cfg.get("architecture", [5, 6, 4, 2]),
        nn_activation    = nn_cfg.get("activation", "tanh"),
    )

    # -- Resume --
    if args.resume:
        metadata = ga.load_checkpoint(output_dir)
        if metadata:
            print(f"  [OK] Da tai checkpoint tu gen {metadata.get('generation', '?')}")
            print(f"       Best fitness: {metadata.get('best_fitness', '?')}")
            print(f"       Saved at: {metadata.get('timestamp', '?')}")
        else:
            print(f"  [--] Khong tim thay checkpoint tai {output_dir}, khoi tao moi.")

    renderer = None
    hud = None
    controls = None
    if not headless:
        from src.rendering.renderer import Renderer
        from src.rendering.hud import HUD
        from src.ui.controls import UIControls
        w = ren_cfg.get("width", 1280)
        h = ren_cfg.get("height", 720)
        renderer = Renderer(
            width        = w,
            height       = h,
            fps          = ren_cfg.get("fps", 60),
            show_sensors = ren_cfg.get("show_sensors", True),
            show_trails  = ren_cfg.get("show_trails", True),
            show_checkpoints = ren_cfg.get("show_checkpoints", True),
        )
        hud = HUD(screen_width=w, screen_height=h)
        hud.set_algorithm("GA")
        controls = UIControls(screen_width=w, screen_height=h)
        controls.speed = speed

    print(f"\n{'='*55}")
    print(f"  Deep Learning Cars -- Genetic Algorithm")
    print(f"  Track: {track_name} | Pop: {ga.population_size} | Gens: {max_gens}")
    print(f"  NN: {nn_cfg.get('architecture', [5,6,4,2])} | Mode: {'headless' if headless else 'render'}")
    if args.resume:
        print(f"  Resume: gen {ga.generation} | Save every: {save_every}")
    print(f"{'='*55}\n")

    # -- MLflow Tracking --
    with MLflowTracker(experiment_name="dl-cars-ga") as tracker:
        tracker.log_params({
            "population_size": ga.population_size,
            "mutation_rate": ga.mutation_rate,
            "architecture": nn_cfg.get("architecture"),
            "track": track_name
        })

        # -- Training loop --
        for gen in range(max_gens):
            if headless:
                fitness_scores, alive_count = run_headless(ga, track)
            else:
                res = run_with_render(ga, track, renderer, hud, controls)
                if res is None:
                    print("\n  Simulation stopped by user.")
                    break
                fitness_scores, alive_count = res

            # Cap nhat HUD chart data
            if hud:
                hud.update({
                    "current_best": max(fitness_scores),
                    "mean_fitness": float(np.mean(fitness_scores)),
                })

            print(ga.log_generation(fitness_scores, alive_count))
            ga.evolve(fitness_scores)

            tracker.log_metrics({
                "best_fitness": max(fitness_scores),
                "mean_fitness": float(np.mean(fitness_scores))
            }, step=ga.generation)

            if save_every > 0 and ga.generation % save_every == 0:
                path = ga.save_checkpoint(output_dir)
                tracker.save_artifact(path)
                print(f"  [SAVE] Checkpoint saved at gen {ga.generation} -> {path}")

        if ga.best_fitness_history:
            path = ga.save_checkpoint(output_dir)
            tracker.save_artifact(path)
            print(f"\n  [SAVE] Final checkpoint saved -> {path}")
            print(f"  Training complete. Best fitness: {max(ga.best_fitness_history):.1f}")
        else:
            print("\n  Training complete (no generations run).")

    if renderer:
        renderer.quit()


# ============================================================================
#  PPO MODE
# ============================================================================

def run_ppo(args, cfg):
    """Training loop cho PPO mode."""
    from src.core.ppo_agent import PPOAgent
    from src.simulation.reward import RewardCalculator

    ppo_cfg = cfg.get("ppo", {})
    rew_cfg = cfg.get("reward", {})
    trk_cfg = cfg.get("track", {})
    trn_cfg = cfg.get("training", {})
    ren_cfg = cfg.get("rendering", {})
    sim_cfg = cfg.get("simulation", {})
    per_cfg = cfg.get("perception", {})

    headless   = args.headless or trn_cfg.get("headless", False)
    track_name = args.track or trk_cfg.get("default", "oval")
    output_dir = trn_cfg.get("output_dir", "./checkpoints")
    save_every = trn_cfg.get("save_best_every", 10)

    total_episodes    = args.generations or ppo_cfg.get("total_episodes", 500)
    max_steps         = ppo_cfg.get("max_steps_per_episode", 3000)
    update_every      = ppo_cfg.get("update_every", 2048)
    
    is_cnn = (per_cfg.get("mode", "sensor") == "pixel")

    # -- Init track --
    track = Track.from_preset(track_name, trk_cfg.get("track_width", 56))
    track.center(ren_cfg.get("width", 1280), ren_cfg.get("height", 720))

    # -- Camera & Renderer (cho CNN) --
    camera = None
    renderer = None
    hud = None
    controls = None
    
    # Neu can render cho nguoi xem, hoac can ve ra de camera chup anh
    if not headless or is_cnn:
        from src.rendering.renderer import Renderer
        w = ren_cfg.get("width", 1280)
        h = ren_cfg.get("height", 720)
        renderer = Renderer(
            width        = w,
            height       = h,
            fps          = ren_cfg.get("fps", 60),
            show_sensors = ren_cfg.get("show_sensors", True),
            show_trails  = ren_cfg.get("show_trails", True),
            show_checkpoints = ren_cfg.get("show_checkpoints", True),
            headless     = headless,
        )
        if not headless:
            from src.rendering.hud import HUD
            from src.ui.controls import UIControls
            hud = HUD(screen_width=w, screen_height=h)
            hud.set_algorithm("PPO")
            controls = UIControls(screen_width=w, screen_height=h)
        
    if is_cnn:
        from src.simulation.camera import CameraSensor
        camera = CameraSensor(
            capture_size = per_cfg.get("capture_size", 84),
            frame_stack  = per_cfg.get("frame_stack", 4),
            grayscale    = per_cfg.get("grayscale", True)
        )
        obs_dim = camera.observation_shape[0]  # so channels (vi du: 4)
    else:
        obs_dim = sim_cfg.get("n_sensors", 5) + 1  # +1 cho compass (angle_diff)

    # -- Init PPO agent --
    agent = PPOAgent(
        obs_dim       = obs_dim,
        action_dim    = 2,
        hidden_sizes  = ppo_cfg.get("hidden_sizes", [64, 64]),
        is_cnn        = is_cnn,
        lr            = ppo_cfg.get("lr", 3e-4),
        gamma         = ppo_cfg.get("gamma", 0.99),
        gae_lambda    = ppo_cfg.get("gae_lambda", 0.95),
        clip_epsilon  = ppo_cfg.get("clip_epsilon", 0.2),
        epochs        = ppo_cfg.get("epochs", 10),
        batch_size    = ppo_cfg.get("batch_size", 64),
        entropy_coeff = ppo_cfg.get("entropy_coeff", 0.01),
        value_coeff   = ppo_cfg.get("value_coeff", 0.5),
        max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5),
    )

    # -- Init reward calculator --
    reward_calc = RewardCalculator.from_config(rew_cfg)

    # -- Resume --
    ckpt_path = Path(output_dir) / "ppo_model.pt"
    if args.resume and ckpt_path.exists():
        agent.load(str(ckpt_path))
        print(f"  [OK] Da tai PPO checkpoint: ep {agent.episode_count}, steps {agent.total_steps}")
    elif args.resume:
        print(f"  [--] Khong tim thay PPO checkpoint tai {ckpt_path}, khoi tao moi.")

    # -- Init dummy car (PPO dung 1 xe duy nhat) --
    dummy_nn = NeuralNetwork(architecture=[5, 4, 2], activation="tanh") # Dummy
    car = Car(
        nn=dummy_nn,
        start_x=track.start_x,
        start_y=track.start_y,
        start_angle=track.start_angle,
    )

    print(f"\n{'='*55}")
    print(f"  Deep Learning Cars -- PPO Agent")
    print(f"  Track: {track_name} | Episodes: {total_episodes}")
    if is_cnn:
        print(f"  Network: CNNActorCritic | Vision: {per_cfg.get('capture_size')}px")
    else:
        print(f"  Network: FC{ppo_cfg.get('hidden_sizes', [64,64])} | Vision: Sensors")
    print(f"  Mode: {'headless' if headless else 'render'} | lr: {ppo_cfg.get('lr', 3e-4)}")
    print(f"{'='*55}\n")

    # -- MLflow Tracking --
    with MLflowTracker(experiment_name="dl-cars-ppo") as tracker:
        tracker.log_params({
            "is_cnn": is_cnn,
            "track": track_name,
            "lr": ppo_cfg.get("lr", 3e-4),
            "gamma": ppo_cfg.get("gamma", 0.99),
            "clip_epsilon": ppo_cfg.get("clip_epsilon", 0.2),
            "epochs": ppo_cfg.get("epochs", 10),
            "batch_size": ppo_cfg.get("batch_size", 64),
        })

        # -- Training loop --
        global_steps = agent.total_steps

        for ep in range(total_episodes):
            car.gym_reset(track.start_x, track.start_y, track.start_angle)
            reward_calc.reset()
            episode_reward = 0.0
            
            # Lay observation dau tien
            if is_cnn:
                camera.reset()
                renderer.draw_frame([car], track, {}) # Ve ra man hinh an
                obs = camera.observe(renderer.screen, car)
            else:
                obs = car.get_observation(track)

            for step_i in range(max_steps):
                # Select action
                action, log_prob, value = agent.select_action(np.array(obs))

                # Step environment (vat ly xe di chuyen)
                next_obs_sensor, base_reward, done, info = car.gym_step(action, track)

                # Shaped reward
                reward = reward_calc.compute(car, track, done)
                episode_reward += reward

                # Neu co CNN thi chup anh lai sau khi xe di chuyen
                if is_cnn:
                    # Ve hinh anh hien tai cua xe tren track
                    renderer.draw_frame([car], track, {})
                    next_obs = camera.observe(renderer.screen, car)
                else:
                    next_obs = next_obs_sensor

                # Store transition
                agent.buffer.store(
                    state=np.array(obs),
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                )

                global_steps += 1
                obs = next_obs

                # Render len man hinh cho nguoi dung (neu khong phai headless)
                if not headless and not done:
                    import pygame
                    events = pygame.event.get()
                    if controls and controls.handle_events(events):
                        print("\n  Simulation stopped by user.")
                        if agent.episode_rewards:
                            Path(output_dir).mkdir(parents=True, exist_ok=True)
                            agent.save(str(ckpt_path))
                            tracker.save_artifact(str(ckpt_path))
                            print(f"  [SAVE] PPO model saved -> {ckpt_path}")
                        renderer.quit()
                        return

                    if controls and controls.paused:
                        # Ve frame hien tai + HUD + Controls
                        renderer.draw_frame([car], track, {})
                        if hud:
                            hud.draw(renderer.screen, {}, 1 if car.alive else 0, 1)
                        controls.draw(renderer.screen)
                        pygame.display.flip()
                        renderer.clock.tick(renderer.fps)
                        continue

                    stats = {
                        "generation": agent.episode_count + 1,
                        "current_best": episode_reward,
                        "all_time_best": max(agent.episode_rewards + [episode_reward]) if agent.episode_rewards else episode_reward,
                    }
                    renderer.draw_frame([car], track, stats)
                    if hud:
                        hud.draw(renderer.screen, stats, 1 if car.alive else 0, 1)
                    if controls:
                        controls.draw(renderer.screen)
                    pygame.display.flip()
                    renderer.clock.tick(renderer.fps)

                # Update policy khi du data
                if len(agent.buffer) >= update_every:
                    update_stats = agent.update()
                    # Log update stats (loss, entropy)
                    tracker.log_metrics(update_stats, step=global_steps)

                if done:
                    break

            # Log episode + update HUD
            agent.total_steps = global_steps
            log_str = agent.log_episode(episode_reward)
            print(log_str)

            if hud:
                hud.update({
                    "current_best": episode_reward,
                    "mean_fitness": float(np.mean(agent.episode_rewards[-100:])) if agent.episode_rewards else episode_reward,
                })
            
            tracker.log_metrics({
                "episode_reward": episode_reward,
                "episode_steps": step_i + 1
            }, step=ep)

            # Auto-save
            if save_every > 0 and (agent.episode_count % save_every == 0):
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                agent.save(str(ckpt_path))
                tracker.save_artifact(str(ckpt_path))
                print(f"  [SAVE] PPO model saved at ep {agent.episode_count} -> {ckpt_path}")

        # Final update voi data con lai trong buffer
        if len(agent.buffer) > 0:
            agent.update()

        # Final save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        agent.save(str(ckpt_path))
        tracker.save_artifact(str(ckpt_path))
        best_reward = max(agent.episode_rewards) if agent.episode_rewards else 0.0
        print(f"\n  [SAVE] Final PPO model saved -> {ckpt_path}")
        print(f"  Training complete. Best episode reward: {best_reward:.1f}")

    if renderer:
        renderer.quit()


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Cars")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--algorithm",   default=None, help="ga | ppo")
    parser.add_argument("--track",       default=None, help="oval | figure8 | city")
    parser.add_argument("--headless",    action="store_true")
    parser.add_argument("--generations", type=int, default=None,
                        help="Generations (GA) or Episodes (PPO)")
    parser.add_argument("--speed",       type=int, default=1,
                        help="Physics steps per render frame (1-10)")
    parser.add_argument("--resume",      action="store_true",
                        help="Tiep tuc huan luyen tu checkpoint da luu")
    args = parser.parse_args()

    # -- Load config --
    cfg = load_config(args.config)
    apply_config(cfg)

    algorithm = args.algorithm or cfg.get("algorithm", "ga")

    if algorithm == "ppo":
        run_ppo(args, cfg)
    else:
        run_ga(args, cfg)


if __name__ == "__main__":
    main()
