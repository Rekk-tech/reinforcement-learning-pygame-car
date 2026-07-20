import pygame
from src.simulation.track import PRESET_TRACKS, Track

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    
    PRESET_TRACKS["map_straight"] = [
        (200, 100, 60), (600, 100, 60),
        (650, 120, 60), (650, 180, 60),
        (600, 200, 60), (200, 200, 60),
        (150, 180, 60), (150, 120, 60)
    ]
    PRESET_TRACKS["map_u_turn"] = [
        (200, 100, 60), (600, 100, 60),
        (650, 150, 80), (600, 200, 60),
        (200, 200, 60), (150, 150, 80)
    ]
    PRESET_TRACKS["map_zigzag"] = [
        (100, 100, 60), (300, 100, 60), 
        (400, 200, 70), (500, 200, 70), 
        (600, 100, 60), (700, 100, 60),
        (750, 250, 80),
        (700, 400, 60), (600, 400, 60),
        (500, 300, 70), (400, 300, 70),
        (300, 400, 60), (100, 400, 60),
        (50, 250, 80)
    ]

    tracks = [
        Track(PRESET_TRACKS["map_straight"], name="map_straight"),
        Track(PRESET_TRACKS["map_u_turn"], name="map_u_turn"),
        Track(PRESET_TRACKS["map_zigzag"], name="map_zigzag")
    ]
    
    for t in tracks:
        t.center(800, 600)
        
    current = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current = (current + 1) % len(tracks)
                    
        screen.fill((20, 20, 20))
        tracks[current].draw(screen)
        
        font = pygame.font.SysFont(None, 36)
        text = font.render(tracks[current].name, True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
