import pygame
import os

def init():
    pygame.init()
    os.environ["DISPLAY"] = ":0"
    pygame.display.init()
    win = pygame.display.set_mode((100,100))

def getKey(keyName):
    ans = False
    for eve in pygame.event.get():pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame,'K_{}'.format(keyName))
    if keyInput [myKey]:
        ans = True
    pygame.display.update()
    
    return ans

def main():
    if getKey('w'):
        print('Forward function is selected')
    if getKey('s'):
        print('Backward function is selected')
    if getKey('a'):
        print('Left function is selected')
    if getKey('d'):
        print('Right function is selected')

if __name__ == '__main__':
    init()
    while True:
            main()
