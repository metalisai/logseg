#!/bin/bash

while read id; do
  outfile="responses/response_$id.json"

  echo "Fetching ID: $id"

  if [ -f "$outfile" ]; then
    echo "Skipping ID $id (already downloaded)"
    continue
  fi

  curl -s "https://register.metsad.ee/portaal/api/rest/eraldis/detail/$id" -o "$outfile"
  status=$(curl -s -w "%{http_code}" -o "$outfile" "https://register.metsad.ee/portaal/api/rest/eraldis/detail/$id")
  echo "Status code: $status for ID $id"
done < ids.txt
