N_TASKS = 1000

test:
	pytest src

all_exp_no_color_aware:
	for augmentation in rotation deformation cropping vertical_flipping horizontal_flipping color_jiter solarize; do\
		python -m scripts.main --number-of-tasks $(N_TASKS) --basic-augmentation $${augmentation} ;\
	done ;\
	python -m scripts.main --number-of-tasks $(N_TASKS) --style-transfer-augmentation ;\
	python -m scripts.main --number-of-tasks $(N_TASKS) ;\
	python -m scripts.main --number-of-tasks $(N_TASKS) --basic-augmentation rotation,deformation,cropping,vertical_flipping,horizontal_flipping,color_jiter,solarize ;\

all_exp_color_aware:
	for augmentation in rotation deformation cropping vertical_flipping horizontal_flipping color_jiter solarize; do\
		python -m scripts.main --number-of-tasks $(N_TASKS) --color-aware --basic-augmentation $${augmentation} ;\
	done ;\
	python -m scripts.main --number-of-tasks $(N_TASKS) --color-aware --style-transfer-augmentation ;\
	python -m scripts.main --number-of-tasks $(N_TASKS) --color-aware ;\
	python -m scripts.main --number-of-tasks $(N_TASKS) --color-aware --basic-augmentation rotation,deformation,cropping,vertical_flipping,horizontal_flipping,color_jiter,solarize ;\

all_exp:
	for task_sampler in --color-aware --no-color-aware; do\
	    for model in prototypical tim finetune; do\
		    for augmentation in rotation deformation cropping vertical_flipping horizontal_flipping color_jiter solarize; do\
		        python -m scripts.main --number-of-tasks $(N_TASKS) $${task_sampler} --few-shot-method $${model} --basic-augmentation $${augmentation} ;\
		    done ;\
			python -m scripts.main --number-of-tasks $(N_TASKS) $${task_sampler} --few-shot-method $${model} ;\
			python -m scripts.main --number-of-tasks $(N_TASKS) $${task_sampler} --few-shot-method $${model} --style-transfer-augmentation ;\
			python -m scripts.main --number-of-tasks $(N_TASKS) $${task_sampler} --few-shot-method $${model} --style-transfer-augmentation --basic-augmentation rotation,deformation,cropping,vertical_flipping,horizontal_flipping,color_jiter,solarize ;\
			python -m scripts.main --number-of-tasks $(N_TASKS) $${task_sampler} --few-shot-method $${model} --basic-augmentation rotation,deformation,cropping,vertical_flipping,horizontal_flipping,color_jiter,solarize ;\
		done ;\
	done ;\
