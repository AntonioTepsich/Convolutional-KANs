# lint
.PHONY: lint
lint:
	poetry run pysen run lint

# format
.PHONY: format
format:
	poetry run pysen run format

# test
.PHONY: test
test:
	poetry run pytest -s -vv ./tests

# get mnist image
.PHONY: get_mnist
get_mnist:
	cd conv_animation && \
	poetry run python -m get_mnist_image

# get animation
.PHONY: anim
anim:
	cd conv_animation && \
	poetry run python -m image_visualization


# show manim
.PHINY: manim
manim:
	cd conv_animation && \
	poetry run manim -p -qh conv_animation.py ConvAnim


# save mp4 to gif
.PHINY: mp42gif
mp42gif:
	cd conv_animation && \
	poetry run python -m mp42gif