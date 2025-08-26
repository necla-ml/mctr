.PHONY: pull

pull:
	git pull
	git submodule update --init --recursive