.PHONY: install build preview lint clean

install:
	uv sync --group book

build:
	cd book && uv run jupyter book build --execute --html

preview:
	cd book && uv run jupyter book start --execute

lint:
	uv run pre-commit run --all-files

clean:
	rm -rf book/_build
