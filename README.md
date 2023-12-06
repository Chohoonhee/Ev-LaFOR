# Ev-LaFOR (ICCV 2023 Oral)

This repository contains the official PyTorch implementation of the paper "Label-Free Event-based Object Recognition via Joint Learning with
Image Reconstruction from Events" paper (ICCV 2023, Oral).

\[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cho_Label-Free_Event-based_Object_Recognition_via_Joint_Learning_with_Image_Reconstruction_ICCV_2023_paper.pdf)\] 




## Qualitative Results on N-Caltech101 and N-ImageNet100 datasets
<img src="https://github.com/intelpro/CBMNet/raw/main/figure/popcorn.gif" width="100%" height="100%">
<!--
![real_event_045_resized](/figure/video_results_real_event3.gif "real_event_045_resized")
-->

### Quantitative results on N-Caltech101 and N-ImageNet100 datasets
<img src="https://github.com/intelpro/CBMNet/raw/main/figure/Quantitative_eval_ERF_x170FPS.png" width="60%" height="60%">



## Requirements
* CLIP (https://github.com/openai/CLIP)


## Training & Test Code

Train & Test on N-Caltech 101 Dataset

```bash
    $ python run_samples.py  --model_name ours --ckpt_path pretrained_model/ours_weight.pth --save_output_dir ./output --image_number 0
```

Train & Test on N-ImageNet 100 Dataset

```bash
    $ python run_samples.py  --model_name ours_large --ckpt_path pretrained_model/ours_large_weight.pth --save_output_dir ./output --image_number 0
```

## Reference
> Hoonhee Cho*, Hyeonseong Kim*, Yujeong Chae, and Kuk-Jin Yoon "Label-Free Event-based Object Recognition via Joint Learning with Image Reconstruction from Events", In _ICCV_, 2023.
```bibtex
@inproceedings{cho2023label,
  title={Label-Free Event-based Object Recognition via Joint Learning with Image Reconstruction from Events},
  author={Cho, Hoonhee and Kim, Hyeonseong and Chae, Yujeong and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19866--19877},
  year={2023}
}
```

##

## Contact
If you have any question, please send an email to hoonhee cho (gnsgnsgml@kaist.ac.kr)

## License
The project codes and datasets can be used for research and education only. 
