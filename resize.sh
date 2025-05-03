find images_falsecolor -type f -iname "*.tiff" -exec sh -c '
  for img; do
    dest="images_falsecolor_2242/$(dirname "$img" | sed "s|/images|/resized_images|")"
    mkdir -p "$dest"
    convert "$img" -resize 224x224 -strip -alpha off "$dest/$(basename "$img")"
  done
' sh {} +
