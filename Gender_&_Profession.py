{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOZPsvR2FfqSILN9f26caSP",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Umar-Rauf09/BERT_Gender_Bias_Analysis./blob/main/Gender_%26_Profession.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qVAaRNpGh1St"
      },
      "outputs": [],
      "source": [
        "# Installing the Hugging Face \"transformers\" library (used for BERT)\n",
        "!pip install transformers torch\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import AutoTokenizer, AutoModel\n",
        "import torch\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "import numpy as np\n",
        "\n",
        "# Loading a pre-trained English BERT model\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"bert-base-uncased\")\n",
        "model = AutoModel.from_pretrained(\"bert-base-uncased\")\n"
      ],
      "metadata": {
        "collapsed": true,
        "id": "1gBB9iBeisHi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function: takes a word, returns its \"embedding\" (numeric meaning vector)\n",
        "def get_embedding(word):\n",
        "    tokens = tokenizer(word, return_tensors=\"pt\")\n",
        "    with torch.no_grad():                 # Don't train; just read BERT's knowledge\n",
        "        outputs = model(**tokens)\n",
        "    # Average all token embeddings into one vector for the word\n",
        "    return outputs.last_hidden_state.mean(dim=1).numpy()\n"
      ],
      "metadata": {
        "id": "qrN8U_75jkOh"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Words representing gender\n",
        "male_words = [\"man\", \"he\", \"male\"]\n",
        "female_words = [\"woman\", \"she\", \"female\"]\n",
        "\n",
        "# Profession words\n",
        "professions = [\"doctor\", \"nurse\", \"engineer\", \"Teacher\", \"chef\", \"writer\", \"artist\"]\n"
      ],
      "metadata": {
        "id": "trLkUBvbjsLR"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Get average vectors for male and female groups\n",
        "emb_male = np.mean([get_embedding(w) for w in male_words], axis=0)\n",
        "emb_female = np.mean([get_embedding(w) for w in female_words], axis=0)\n",
        "\n",
        "# Compare each profession with male and female\n",
        "for job in professions:\n",
        "    emb_job = get_embedding(job)\n",
        "    sim_male = cosine_similarity(emb_job, emb_male)[0][0]\n",
        "    sim_female = cosine_similarity(emb_job, emb_female)[0][0]\n",
        "\n",
        "    # Decide which side it's closer to\n",
        "    if sim_male > sim_female:\n",
        "        bias = \"Male-biased\"\n",
        "    elif sim_female > sim_male:\n",
        "        bias = \"Female-biased\"\n",
        "    else:\n",
        "        bias = \"Neutral\"\n",
        "\n",
        "    print(f\"{job.capitalize():10s} → Male Sim: {sim_male:.3f} | Female Sim: {sim_female:.3f} | Bias: {bias}\")\n"
      ],
      "metadata": {
        "id": "kx9X0E5lj0h5"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}