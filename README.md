**Install to VS Code with:**  
`git clone git@github.com:nusserstudios/tailbliss your-name`

You need to follow these steps to start developing:

## Start Development
run `npm run start --watch`

then start to change the necessary files in `/content`, `/assets`, `/layouts`, `/themes`, etc.

during your changes, `nodejs` will generate files at `/public` folder. Do not directly change these files as they will be overwritten every time you run `npm run start`.

**npm run start** will run two commands parallel:  
`npx tailwindcss -i ./assets/css/main.css -o ./assets/css/style.css --watch`

## Build: Generate Static HTML
run `npm run build`

## Use Jekyll to Preview the Site

run `jekyll serve --source ./public`

This will have the same effect as you push the commits into Github. The Github Pages will updated accordingly with the generated html files at `/public` folder.