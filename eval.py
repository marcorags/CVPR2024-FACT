import torch
from tqdm import tqdm
from .configs.utils import get_cfg_defaults
from .utils.dataset import create_dataset, DataLoader
from .utils import utils
from .utils.evaluate import Checkpoint, Video
from .utils.train_tools import save_results


for dataset_name, n_splits in [
        ['fsjump', 1]
    ]:
    print(dataset_name)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(f'./CVPR2024-FACT/configs/{dataset_name}.yaml')

    ckpts = []
    for split in range(1, n_splits+1):
        cfg.split = f"split{split}"
        dataset, test_dataset = create_dataset(cfg)

        # FACT Model
        from .models.blocks import FACT 
        model = FACT(cfg, dataset.input_dimension, dataset.nclasses)

        # Find the path of the best checkpoint
        best_ckpt_path = f'./CVPR2024-FACT/log/{dataset_name}/{dataset_name}/0/best_ckpt.gz'
        best_ckpt = Checkpoint.load(best_ckpt_path)

        # Load weights from the .net file corresponding to the best iteration
        weights_path = f'./CVPR2024-FACT/log/{dataset_name}/{dataset_name}/0/ckpts/network.iter-{best_ckpt.iteration}.net'
        weights = torch.load(weights_path, map_location='cpu')
        
        # best_ckpt_path = f'./CVPR2024-FACT/log/{dataset_name}/{dataset_name}/0/best_ckpt.gz'
        # best_ckpt = Checkpoint.load(best_ckpt_path)  # Load entire checkpoint
        # print("best_ckpt --> ", best_ckpt)
        # weights = best_ckpt.videos

        # weights = f'/home/disi/siv/SIV_UniTN_TAS_project/log/fsjump/fsjump/0/best_ckpt.gz'
        # weights = f'./ckpts/{dataset_name}/split{split}-weight.pth'
        # weights = torch.load(weights, map_location='cpu')
        if 'frame_pe.pe' in weights:
            del weights['frame_pe.pe']
        model.load_state_dict(weights, strict=False)
        model.eval().cuda()


        ckpt = Checkpoint(-1, bg_class=([] if cfg.eval_bg else dataset.bg_class))
        loader  = DataLoader(test_dataset, 1, shuffle=False)
        for vname, batch_seq, train_label_list, eval_label in tqdm(loader):
            seq_list = [ s.cuda() for s in batch_seq ]
            train_label_list = [ s.cuda() for s in train_label_list ]
            video_saves = model(seq_list, train_label_list, compute_loss=False)
            save_results(ckpt, vname, eval_label, video_saves)

        ckpt.compute_metrics()
        ckpts.append(ckpt)

    print(utils.easy_reduce([c.metrics for c in ckpts]))
