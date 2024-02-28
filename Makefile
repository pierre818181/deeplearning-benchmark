build-and-push-image:
	@echo "Building benchmark image"
	sudo docker build --no-cache -t pierre781/runpod_benchmark:latest .
	sudo docker push pierre781/runpod_benchmarks:latest
