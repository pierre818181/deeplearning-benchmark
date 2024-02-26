build-and-push-heavy:
	@echo "Building heavy"
	sudo docker build --no-cache -t pierre781/runpod_benchmarks:heavy .
	sudo docker push pierre781/runpod_benchmarks:heavy

build-and-push-lite:
	@echo "Building lite"
	sudo docker build --no-cache -t pierre781/runpod_benchmarks:lite .
	sudo docker push pierre781/runpod_benchmarks:lite