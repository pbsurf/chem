import sys
import ctypes

# git clone https://github.com/marcusva/py-sdl2.git
sys.path.append("./py-sdl2")
from sdl2 import *

def main():
  SDL_Init(SDL_INIT_VIDEO)
  window = SDL_CreateWindow(b"Hello World",
      SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 592, 460, SDL_WINDOW_SHOWN)
  windowsurface = SDL_GetWindowSurface(window)

  #image = SDL_LoadBMP(b"exampleimage.bmp")
  #SDL_BlitSurface(image, None, windowsurface, None)

  SDL_UpdateWindowSurface(window)
  #SDL_FreeSurface(image)

  running = True
  event = SDL_Event()
  while running:
    while SDL_WaitEvent(ctypes.byref(event)) != 0:
      if event.type == SDL_QUIT:
        running = False
        break
      elif event.type == SDL_KEYDOWN:
        if event.key.keysym.sym == SDLK_ESCAPE:
          running = False
          break
      elif event.type == SDL_FINGERMOTION:
        print "Finger %d moved to (%f, %f)" % (event.tfinger.fingerId, event.tfinger.x, event.tfinger.y)

  SDL_DestroyWindow(window)
  SDL_Quit()
  return 0

if __name__ == "__main__":
  sys.exit(main())
