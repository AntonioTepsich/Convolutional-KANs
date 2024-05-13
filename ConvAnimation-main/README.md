# ConvAnimation
The animation of Convolutional

# Project Cards
https://github.com/i13abe/ConvAnimation/projects/1?add_cards_query=is%3Aopen

# How to use
## Install poetry
Python のpackage管理用にPoetryを利用する.<br>
see: https://python-poetry.org/docs/#installation

## Install libraries
poetryでパッケージをインストールする.
```sh
$ poetry install
```

pytorchのみ自前で取ってくる.<br>
以下のリンクより適した環境のものを取得.<br>
https://pytorch.org/

## Install manim
私はWindowsのchocolateyで導入した.
```sh
choco install manimce
```

さらにpython pipでも取得する
```sh
poetry run pip install manim
```

そのほかの環境は以下のページより適切な手段で取得する.<br>
see: https://docs.manim.community/en/stable/installation.html

## Install LaTex
Windowsの場合MiKTeXをインストールする.<br>
see:https://self-development.info/miktex%e3%81%aewindows%e3%81%b8%e3%81%ae%e3%82%a4%e3%83%b3%e3%82%b9%e3%83%88%e3%83%bc%e3%83%ab/

## Run and get animation
```sh
$ make manim
```