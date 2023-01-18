test:
	pytest src

all_exp_no_color_aware:
	for augmentation in rotation deformation cropping vertical_flipping horizontal_flipping color_jiter solarize; do\
		python -m scripts.main --number-of-tasks 1000 --basic-data-augmentation $${augmentation} ;\
	done ;\
	python -m scripts.main --number-of-tasks 1000 --style-transfer-augmentation ;\
	python -m scripts.main --number-of-tasks 1000 ;\

all_exp_color_aware:
	for augmentation in rotation deformation cropping vertical_flipping horizontal_flipping color_jiter solarize; do\
		python -m scripts.main --number-of-tasks 1000 --color-aware --basic-data-augmentation $${augmentation} ;\
	done ;\
	python -m scripts.main --number-of-tasks 1000 --color-aware --style-transfer-augmentation ;\
	python -m scripts.main --number-of-tasks 1000 --color-aware ;\
