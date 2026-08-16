import sys, json, os, pathlib

checkpoint_folder = sys.argv[1]
delete_folders = len(sys.argv) > 2 and sys.argv[2] == "-delete"
print("Delete folders:", delete_folders)

checkpoints = os.listdir(checkpoint_folder)
checkpoints = [x for x in checkpoints if "-" in x]
best_model_checkpoint = ""
best_metric = 0
for checkpoint in checkpoints:
	if "trainer_state.json" in os.listdir(checkpoint_folder + "/" + checkpoint):
		state = json.load(open(checkpoint_folder + "/" + checkpoint + "/trainer_state.json", "r"))
		if state["best_metric"] > best_metric:
			best_metric = state["best_metric"]
			best_model_checkpoint = state["best_model_checkpoint"].split("/")[-1]
print("Best checkpoint:", best_model_checkpoint, best_metric)
fd = os.open(checkpoint_folder, os.O_RDONLY)
os.symlink(best_model_checkpoint, "best", dir_fd=fd)

if delete_folders:
	for checkpoint in checkpoints:
		if checkpoint != best_checkpoint:
			path = pathlib.Path(checkpoint_folder + "/" + checkpoint)
			path.unlink()
