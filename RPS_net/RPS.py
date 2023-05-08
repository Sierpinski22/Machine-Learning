import pygame
from random import randint
from network import *

loaded = False
only_net = False
side, sep = 10, 5
size = 600
pygame.init()
screen = pygame.display.set_mode((size * 2 + sep if not only_net else size, size))

w, h = screen.get_size()
cols, rows = int(w / side / 2) if not only_net else int(w / side), int(h / side)

tab = [[randint(0, 2) for _ in range(cols)] for _ in range(rows)]
old_tab = tab.copy()

iteration = 0
threshold = 3

# ------------------------- #
net, optimizer = build_net()
# ------------------------- #



def update(ot):
    new_t = [[_ for _ in range(cols)] for _ in range(rows)]
    for y_, ro in enumerate(ot):
        for x_, co in enumerate(ro):
            n = 0
            neighbors = []
            for y1 in range(y_ - 1, y_ + 2):
                for x1 in range(x_ - 1, x_ + 2):
                    neighbors.append(ot[(y1 + rows) % rows][(x1 + cols) % cols] / 2)
                    if y1 != y_ or x1 != x_:
                        xc = (x1 + cols) % cols
                        yc = (y1 + rows) % rows
                        n += 1 if (ot[y_][x_] == 0 and ot[yc][xc] == 1) or (ot[y_][x_] == 1 and ot[yc][xc] == 2) or (
                                ot[y_][x_] == 2 and ot[yc][xc] == 0) else 0

            if n >= 3:
                if ot[y_][x_] == 0:
                    new_t[y_][x_] = 1
                elif ot[y_][x_] == 1:
                    new_t[y_][x_] = 2
                else:
                    new_t[y_][x_] = 0
            else:
                new_t[y_][x_] = ot[y_][x_]

            s = train(neighbors, new_t[y_][x_] / 2, net, optimizer)

            c = (255, 0, 0) if new_t[y_][x_] == 0 else (0, 0, 0)
            c = (0, 255, 0) if new_t[y_][x_] == 1 else c
            c = (0, 0, 255) if new_t[y_][x_] == 2 else c

            # divisione in tre, ma colore basato sulla differenza dal punto di interesse

            c_ = s % (1 / 3)
            # print(s)

            # c1 = (int(255 - 255 * s * 3), 0, 0) if s <= 1 / 3 else (0, 0, 0)
            # c1 = (0, int(255 - 255 * abs(0.5 - s) * 3), 0) if s > 1 / 3 else c1
            # c1 = (0, 0, int(255 - 255 * (1 - s) * 3)) if s > 2 / 3 else c1

            c1 = (int(255 - s * 255), 0, 0) if s <= 1 / 3 else (0, 0, 0)
            c1 = (0, int(255 - abs(0.5 - s) * 2 * 255), 0) if 1 / 3 < s <= 2 / 3 else c1
            c1 = (0, 0, int(255 - (1 - s) * 255)) if s > 2 / 3 else c1

            pygame.draw.rect(screen, c1, (x_ * side, y_ * side, side, side))
            pygame.draw.rect(screen, c, (x_ * side + int(w / 2 + sep), y_ * side, side, side))

    return new_t.copy()


while True:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break

    tab = update(old_tab)
    old_tab = tab.copy()

    pygame.display.update()
    iteration += 1
    pygame.image.save(screen, 'Generated/' + 'epoch_{0:0=2d}.jpg'.format(iteration))
