---
title: AerialSegmentation
emoji: 🌖
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 6.26.0
python_version: '3.11'
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Local Docker Run

If you want to test or host this on your laptop without burning Hugging Face quota, use Docker.

```bash
docker build -t aerialsegmentation .
docker run --rm -p 7860:7860 aerialsegmentation
```

Then open `http://localhost:7860`.

Native Windows `python app.py` is not the easy path for this project because Detectron2 needs a Linux-style build setup. Docker or WSL is the practical local option.
