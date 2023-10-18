# (c) Alex Spirin 2022-2023
from PIL import Image, ImageDraw

def hstack(images):
  if isinstance(images[0], str):
    images = [Image.open(image).convert('RGB') for image in images]
  widths, heights = zip(*(i.size for i in images))
  for image in images:
    draw = ImageDraw.Draw(image)
    draw.rectangle(((0, 00), (image.size[0], image.size[1])), outline="black", width=3)
  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  return new_im

def vstack(images):
  if isinstance(next(iter(images)), str):
    images = [Image.open(image).convert('RGB') for image in images]
  widths, heights = zip(*(i.size for i in images))

  total_height = sum(heights)
  max_width = max(widths)

  new_im = Image.new('RGB', (max_width, total_height))

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]
  return new_im