import pygame
from random import random
from network import *


loaded = False
only_net = False
side, sep = 10, 5
size = 600
pygame.init()
screen = pygame.display.set_mode((size * 2 + sep if not only_net else size, size))

w, h = screen.get_size()
cols, rows = int(w / side / 2) if not only_net else int(w / side), int(h / side)

tab = [[1 if random() < 0.5 else 0 for _ in range(cols)] for _ in range(rows)]
old_tab = tab.copy()

iteration = 0
# ---------------------------------- #
if not loaded:
    net, optimizer = build_net()
else:
    net = load()
# ---------------------------------- #



def update(ot):
    new_t = [[_ for _ in range(cols)] for _ in range(rows)]
    for y_, ro in enumerate(ot):
        for x_, co in enumerate(ro):
            n = 0
            neighbors = []
            for y1 in range(y_ - 1, y_ + 2):
                for x1 in range(x_ - 1, x_ + 2):
                    state = ot[(y1 + rows) % rows][(x1 + cols) % cols]
                    neighbors.append(state)

                    if x_ != x1 or y_ != y1:  # quinto elemento
                        n += state

            if ot[y_][x_] == 0 and n == 3:
                new_t[y_][x_] = 1
            elif ot[y_][x_] == 0:
                new_t[y_][x_] = 0
            elif ot[y_][x_] == 1 and (n > 3 or n < 2):
                new_t[y_][x_] = 0
            else:
                new_t[y_][x_] = 1


            s = train(neighbors, new_t[y_][x_], net, optimizer) if not loaded else generate(neighbors, net)

            color = int(255 * s)
            pygame.draw.rect(screen, (color, color, color), (x_ * side, y_ * side, side, side))
            if not only_net:
                color = int(255 * new_t[y_][x_])
                pygame.draw.rect(screen, (color, color, color), (x_ * side + int(w / 2) + sep, y_ * side, side, side))

    if iteration == 10 and not loaded:
        print('saved')
        save(net)

    return new_t.copy()




while True:
    screen.fill((255, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break

    tab = update(old_tab)
    old_tab = tab.copy()
    pygame.display.update()
    iteration += 1
    # pygame.image.save(screen, 'Generated/' + 'epoch_{0:0=2d}.jpg'.format(iteration))
