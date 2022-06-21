REMOTE_HOST=gk406
REMOTE_PATH=/home/pvuser/Projects/uw-madison-gi-tract-image-segmentation

.PHONY: watch
watch:
	watchman watch $(CURDIR)
	echo '["trigger", "$(CURDIR)", {"name": "jupytext", "stdin": "NAME_PER_LINE", "append_files": false, "expression": ["match", "*.ipynb", "wholename"], "command": ["xargs", "jupytext", "--to=py:percent"]}]' | watchman -j -
	echo '["trigger", "$(CURDIR)", {"name": "rsync", "stdin": "NAME_PER_LINE", "append_files": false, "command": ["xargs", "-I_", "rsync", "_", "$(REMOTE_HOST):$(REMOTE_PATH)"]}]' | watchman -j -

.PHONY: unwatch
unwatch:
	watchman watch-del $(CURDIR)

.PHONY: create-task
create-task:
	ssh $REMOTE_HOST 'clearml-task --project $(basename $PWD) --name DiNTS --folder $PWD --script DiNTS.py --queue default --requirements requirements.txt --docker nvidia/cuda:11.0-cudnn7-runtime-ubuntu20.04'