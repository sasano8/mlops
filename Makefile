include common/Makefile

all: format test-full

format: format-black format-isort

format-black:
	@echo [black] && poetry run black . -v

format-isort:
	@echo [isort] && poetry run isort --profile black --filter-files .

test:
	@echo [pytest] && poetry run pytest -sv -m "not slow" -x # x -１つエラーが発生したら中断する

test-mypy:
	@poetry run mypy tests/mypy > tests/mypy/result.txt || echo ""

test-full: test-mypy
	@echo [pytest] && poetry run pytest . -sv
