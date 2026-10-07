// Generated from the paper by scripts/refresh_content.py; do not edit.
window.MOL_RESULTS = {
  "metrics": [
    "V*",
    "GPT4V-hard",
    "OCR",
    "Direct attr.",
    "Rel. pos.",
    "MMStar",
    "HRBench8K",
    "HRBench4K",
    "RealWorldQA",
    "NaturalBench",
    "CharXiv"
  ],
  "groups": {
    "clip": {
      "label": "Vicuna-13B + CLIP",
      "rows": [
        {
          "method": "Baseline",
          "scores": [
            71.43,
            52.94,
            50.0,
            65.22,
            93.42,
            30.0,
            36.88,
            43.75,
            51.64,
            68.93,
            23.84
          ]
        },
        {
          "method": "MoL layer",
          "scores": [
            76.89,
            58.82,
            53.33,
            75.65,
            92.11,
            29.53,
            38.0,
            40.62,
            52.35,
            69.71,
            26.25
          ]
        },
        {
          "method": "MoL patch",
          "scores": [
            84.87,
            58.82,
            60.0,
            86.09,
            98.68,
            30.13,
            38.5,
            42.25,
            51.82,
            69.41,
            29.17
          ]
        },
        {
          "method": "MoL hybrid",
          "scores": [
            84.03,
            64.71,
            56.67,
            85.22,
            97.37,
            31.07,
            38.62,
            43.88,
            53.14,
            70.36,
            25.56
          ]
        }
      ],
      "source": "experiments.tex (main backbone table)"
    },
    "dinov2": {
      "label": "Vicuna-13B + DINOv2 w/ Txt",
      "rows": [
        {
          "method": "Baseline",
          "scores": [
            65.55,
            76.47,
            40.0,
            67.83,
            69.74,
            28.87,
            32.75,
            35.38,
            50.46,
            68.29,
            22.92
          ]
        },
        {
          "method": "MoL layer",
          "scores": [
            67.65,
            52.94,
            36.67,
            60.87,
            93.42,
            27.8,
            34.12,
            35.38,
            50.46,
            67.92,
            25.94
          ]
        },
        {
          "method": "MoL patch",
          "scores": [
            76.05,
            64.71,
            40.0,
            74.78,
            94.74,
            28.47,
            33.37,
            34.88,
            51.02,
            66.03,
            17.71
          ]
        },
        {
          "method": "MoL hybrid",
          "scores": [
            84.45,
            70.59,
            50.0,
            86.09,
            98.68,
            30.13,
            36.5,
            39.87,
            52.29,
            67.89,
            26.92
          ]
        }
      ],
      "source": "experiments.tex (main backbone table)"
    },
    "llama": {
      "label": "Llama-3-8B + SigLIP",
      "rows": [
        {
          "method": "Baseline",
          "scores": [
            54.62,
            47.06,
            66.67,
            56.52,
            48.68,
            37.73,
            44.0,
            51.12,
            59.87,
            74.75,
            25.05
          ]
        },
        {
          "method": "MoL patch",
          "scores": [
            52.52,
            41.18,
            43.33,
            60.0,
            47.37,
            37.07,
            40.0,
            48.88,
            58.92,
            75.02,
            26.16
          ]
        },
        {
          "method": "MoL hybrid",
          "scores": [
            63.03,
            47.06,
            76.67,
            70.43,
            50.0,
            33.4,
            42.13,
            45.0,
            45.62,
            72.05,
            26.82
          ]
        }
      ],
      "source": "experiments.tex (main backbone table)"
    },
    "phi": {
      "label": "Phi-1.5-1.3B + SigLIP",
      "rows": [
        {
          "method": "Baseline",
          "scores": [
            42.01,
            23.52,
            26.66,
            35.65,
            61.84,
            34.93,
            32.0,
            36.5,
            34.93,
            67.31,
            4.07
          ]
        },
        {
          "method": "MoL patch",
          "scores": [
            39.92,
            29.41,
            50.0,
            28.7,
            55.26,
            33.6,
            30.63,
            33.12,
            38.82,
            60.86,
            19.65
          ]
        },
        {
          "method": "MoL hybrid",
          "scores": [
            50.84,
            23.52,
            43.33,
            42.6,
            72.36,
            34.93,
            32.0,
            37.25,
            39.74,
            68.0,
            20.46
          ]
        }
      ],
      "source": "experiments.tex (main backbone table)"
    },
    "multi": {
      "label": "Multi-encoder \u00b7 Vicuna-13B",
      "rows": [
        {
          "method": "Interleaved-MoF",
          "scores": [
            82.35,
            58.82,
            53.33,
            82.61,
            98.68,
            31.87,
            35.75,
            44.12,
            57.77,
            67.26,
            26.73
          ]
        },
        {
          "method": "MoL hybrid",
          "scores": [
            90.34,
            82.35,
            66.67,
            92.17,
            98.68,
            32.0,
            37.62,
            43.5,
            52.55,
            69.72,
            31.13
          ]
        }
      ],
      "source": "multi_encoder_table.tex"
    }
  }
};
