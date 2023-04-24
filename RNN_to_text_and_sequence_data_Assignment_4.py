{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
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
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "pBOyL2d3C4IR",
        "outputId": "ffa23b2f-a0ce-4b48-d498-5938ebc488f5"
      },
      "source": [
        "import keras\n",
        "keras.__version__"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'2.4.3'"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 1
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hEf61I44DlTY"
      },
      "source": [
        "Using Pretrained word embedding"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xa7iamIAC_lu",
        "outputId": "acfc92c1-004f-4453-b205-81b1630bc0c7"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Zu82eYFSDKu_"
      },
      "source": [
        "import os, shutil"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qMbOi-i8DTT7",
        "outputId": "9f84a1eb-d84c-435c-95d7-84762799ccfd"
      },
      "source": [
        "imdb_dir = '/content/drive/MyDrive/aclImdb'\n",
        "!ls '/content/drive/MyDrive/aclImdb'"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            " imdbEr.txt   imdb.vocab  'New Text Document.txt'   README   test   train\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cNwOzNAWD5iC"
      },
      "source": [
        "Import data"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4g3oOKYJDXZR"
      },
      "source": [
        "import os\n",
        "\n",
        "train_dir = os.path.join(imdb_dir, 'train')\n",
        "\n",
        "labels = []\n",
        "texts = []\n",
        "\n",
        "for label_type in ['neg', 'pos']:\n",
        "    dir_name = os.path.join(train_dir, label_type)\n",
        "    for fname in os.listdir(dir_name):\n",
        "        if fname[-4:] == '.txt':\n",
        "            f = open(os.path.join(dir_name, fname))\n",
        "            texts.append(f.read())\n",
        "            f.close()\n",
        "            if label_type == 'neg':\n",
        "                labels.append(0)\n",
        "            else:\n",
        "                labels.append(1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aGSj9B0ZD-t0"
      },
      "source": [
        "Tokenize the data"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vxiCuhOlcnPY",
        "outputId": "7628188a-64ca-4cbe-93d4-84bdeb853e82"
      },
      "source": [
        "from keras.preprocessing.text import Tokenizer\n",
        "from keras.preprocessing.sequence import pad_sequences\n",
        "import numpy as np\n",
        "\n",
        "maxlen = 150\n",
        "training_samples = 100\n",
        "validation_samples = 10000\n",
        "max_words = 10000\n",
        "\n",
        "tokenizer = Tokenizer(num_words=max_words)\n",
        "tokenizer.fit_on_texts(texts)\n",
        "sequences = tokenizer.texts_to_sequences(texts)\n",
        "\n",
        "word_index = tokenizer.word_index\n",
        "print('Found %s unique tokens.' % len(word_index))\n",
        "\n",
        "data = pad_sequences(sequences, maxlen=maxlen)\n",
        "\n",
        "labels = np.asarray(labels)\n",
        "print('Shape of data tensor:', data.shape)\n",
        "print('Shape of label tensor:', labels.shape)\n",
        "\n",
        "indices = np.arange(data.shape[0])\n",
        "np.random.shuffle(indices)\n",
        "data = data[indices]\n",
        "labels = labels[indices]\n",
        "\n",
        "x_train = data[:training_samples]\n",
        "y_train = labels[:training_samples]\n",
        "x_val = data[training_samples: training_samples + validation_samples]\n",
        "y_val = labels[training_samples: training_samples + validation_samples]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 88582 unique tokens.\n",
            "Shape of data tensor: (25000, 150)\n",
            "Shape of label tensor: (25000,)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ocuyaLAyENML"
      },
      "source": [
        "import glove 6b"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wMksGnQAdLaq",
        "outputId": "83258859-071a-4583-fad7-e820ba81f51e"
      },
      "source": [
        "glove_dir = '/content/drive/MyDrive/glove.6B'\n",
        "!ls '/content/drive/MyDrive/glove.6B'"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "glove.6B.100d.txt  glove.6B.200d.txt  glove.6B.300d.txt  glove.6B.50d.txt\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "L_TfA-DnETnJ"
      },
      "source": [
        "Preprocess embeddings"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GnwtoxofdQ1o",
        "outputId": "14291ac8-929c-498f-e571-3a68381b11bf"
      },
      "source": [
        "from keras.preprocessing.text import Tokenizer\n",
        "from keras.preprocessing.sequence import pad_sequences\n",
        "import numpy as np\n",
        "\n",
        "embeddings_index = {}\n",
        "f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))\n",
        "for line in f:\n",
        "    values = line.split()\n",
        "    word = values[0]\n",
        "    coefs = np.asarray(values[1:], dtype='float32')\n",
        "    embeddings_index[word] = coefs\n",
        "f.close()\n",
        "\n",
        "print('Found %s word vectors.' % len(embeddings_index))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 400000 word vectors.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "seG6KUF2f_O3"
      },
      "source": [
        "embedding_dim = 100\n",
        "\n",
        "embedding_matrix = np.zeros((max_words, embedding_dim))\n",
        "for word, i in word_index.items():\n",
        "    if i < max_words:\n",
        "        embedding_vector = embeddings_index.get(word)\n",
        "        if embedding_vector is not None:\n",
        "          embedding_matrix[i] = embedding_vector"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sy6K8t5MEbck"
      },
      "source": [
        "Build the model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1KjH8D4XgIwi",
        "outputId": "242e9cd7-0047-4608-a5f6-585ab4a877da"
      },
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Embedding, Flatten, Dense\n",
        "\n",
        "model = Sequential()\n",
        "model.add(Embedding(max_words, embedding_dim, input_length=maxlen))\n",
        "model.add(Flatten())\n",
        "model.add(Dense(32, activation='relu'))\n",
        "model.add(Dense(1, activation='sigmoid'))\n",
        "model.summary()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "embedding (Embedding)        (None, 150, 100)          1000000   \n",
            "_________________________________________________________________\n",
            "flatten (Flatten)            (None, 15000)             0         \n",
            "_________________________________________________________________\n",
            "dense (Dense)                (None, 32)                480032    \n",
            "_________________________________________________________________\n",
            "dense_1 (Dense)              (None, 1)                 33        \n",
            "=================================================================\n",
            "Total params: 1,480,065\n",
            "Trainable params: 1,480,065\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SQAmNMY_ElG7"
      },
      "source": [
        "Load the glove embeddings"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KIqCldrXgQcM"
      },
      "source": [
        "model.layers[0].set_weights([embedding_matrix])\n",
        "model.layers[0].trainable = False"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4LnBArn-EuEW"
      },
      "source": [
        "Train and evaluate"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_D9CHRJSgXc5",
        "outputId": "7a6a1db0-4d13-4332-d5bf-cdbe5d57c2c2"
      },
      "source": [
        "model.compile(optimizer='rmsprop',\n",
        "              loss='binary_crossentropy',\n",
        "              metrics=['acc'])\n",
        "history = model.fit(x_train, y_train,\n",
        "                    epochs=10,\n",
        "                    batch_size=32,\n",
        "                    validation_data=(x_val, y_val))\n",
        "model.save_weights('pre_trained_glove_model.h5')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/10\n",
            "4/4 [==============================] - 2s 296ms/step - loss: 2.2054 - acc: 0.5862 - val_loss: 0.7813 - val_acc: 0.5094\n",
            "Epoch 2/10\n",
            "4/4 [==============================] - 1s 223ms/step - loss: 0.6062 - acc: 0.6652 - val_loss: 0.6983 - val_acc: 0.5461\n",
            "Epoch 3/10\n",
            "4/4 [==============================] - 1s 222ms/step - loss: 0.3793 - acc: 0.8010 - val_loss: 0.9853 - val_acc: 0.5008\n",
            "Epoch 4/10\n",
            "4/4 [==============================] - 1s 237ms/step - loss: 0.1717 - acc: 0.9536 - val_loss: 1.4688 - val_acc: 0.5004\n",
            "Epoch 5/10\n",
            "4/4 [==============================] - 1s 248ms/step - loss: 0.3316 - acc: 0.7941 - val_loss: 0.7181 - val_acc: 0.5312\n",
            "Epoch 6/10\n",
            "4/4 [==============================] - 1s 240ms/step - loss: 0.0996 - acc: 0.9845 - val_loss: 0.9061 - val_acc: 0.5059\n",
            "Epoch 7/10\n",
            "4/4 [==============================] - 1s 247ms/step - loss: 0.0637 - acc: 1.0000 - val_loss: 0.7631 - val_acc: 0.5275\n",
            "Epoch 8/10\n",
            "4/4 [==============================] - 1s 231ms/step - loss: 0.0462 - acc: 1.0000 - val_loss: 0.8088 - val_acc: 0.5412\n",
            "Epoch 9/10\n",
            "4/4 [==============================] - 1s 217ms/step - loss: 0.0248 - acc: 1.0000 - val_loss: 0.7189 - val_acc: 0.5588\n",
            "Epoch 10/10\n",
            "4/4 [==============================] - 1s 239ms/step - loss: 0.0260 - acc: 1.0000 - val_loss: 0.7576 - val_acc: 0.5598\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "O2Vpx5M9FEYh"
      },
      "source": [
        "Plotting"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 545
        },
        "id": "PtmavwvYgeYm",
        "outputId": "64fe3767-1953-4868-8974-53df74a43dcc"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "acc = history.history['acc']\n",
        "val_acc = history.history['val_acc']\n",
        "loss = history.history['loss']\n",
        "val_loss = history.history['val_loss']\n",
        "\n",
        "epochs = range(1, len(acc) + 1)\n",
        "\n",
        "plt.plot(epochs, acc, 'bo', label='Training acc')\n",
        "plt.plot(epochs, val_acc, 'b', label='Validation acc')\n",
        "plt.title('Training and validation accuracy')\n",
        "plt.legend()\n",
        "\n",
        "plt.figure()\n",
        "\n",
        "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
        "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
        "plt.title('Training and validation loss')\n",
        "plt.legend()\n",
        "\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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e/fujBo1iu+++273/saNG0evXr3o3r07y5Ytq/Dni3qaXY1DF0lTV14JCxdWvl5VZGXBXXeV/3irVq3o27cvzz77LIMHDyY3N5ezzz4bM2PChAm0atWKnTt3ctJJJ/H+++/To0ePMrczf/58cnNzWbhwITt27KBXr1707t0bgCFDhnDhhRcC8Nvf/pYHH3yQyy67jEGDBnHaaadx5plnltjWt99+y8iRI3n55Zfp2rUrw4cP59577+XKK68EoE2bNixYsIB77rmHiRMn8sADD5T780U9za5a6CJSp4p3uxTvbnniiSfo1asX2dnZLF68uET3SGmvv/46Z5xxBk2bNmX//fdn0KBBux9btGgRxx9/PN27d2fatGksXry4wno+/PBDOnfuTNeuXQEYMWIEc+bM2f34kCFDAOjdu/fuCb3K88Ybb3DeeecBZU+zO2nSJDZu3EijRo3o06cPU6ZMYfz48XzwwQc0b968wm3HQy10kTRVUUu6Ng0ePJgxY8awYMECCgsL6d27N5988gkTJ07k3XffpWXLlowcObLcaXMrM3LkSGbOnEnPnj2ZOnUqs2fPrlG9RVPw1mT63bFjx3Lqqacya9Ys+vfvz/PPP797mt1nnnmGkSNHctVVVzF8+PAa1aoWuojUqWbNmnHiiScyatSo3a3zzZs3s99++3HAAQewdu1ann322Qq3ccIJJzBz5ky2bt1KQUEBTz/99O7HCgoKOPjgg9m+ffvuKW8BmjdvTkFBwV7b6tatGytXrmT58uUAPProo/zwhz+s1s8W9TS7aqGLSJ0bOnQoZ5xxxu6ul6LpZg8//HA6dOhA//79K3x+r169OOecc+jZsycHHnggffr02f3YbbfdRr9+/Wjbti39+vXbHeLnnnsuF154IZMmTdp9MBSgSZMmTJkyhbPOOosdO3bQp08fLr744mr9XEXXOu3RowdNmzYtMc3uq6++SoMGDTjyyCMZOHAgubm53H777WRkZNCsWbOEXAhD0+eKpBFNn1u/aPpcEZE0pUAXEUkRCnSRNBNVN6tUTXV+Twp0kTTSpEkT1q9fr1BPcu7O+vXradKkSZWep1EuImkkMzOT/Px81q1bF3UpUokmTZqQmZlZpeco0EXSSEZGBp07d466DKkl6nIREUkRCnQRkRShQBcRSREKdBGRFKFAFxFJEZUGupl1MLNXzWyJ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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5V46aXX4FH2N"
      },
      "source": [
        "highest validation accuracy is 56% while training accuracy is 100%. model is overfitting"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "he9DOOB7FdYE"
      },
      "source": [
        "Training the model with an embedding layer"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dh0m8LhigiHd",
        "outputId": "687912c7-c2eb-40d3-e446-c5bf79f21adf"
      },
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Embedding, Flatten, Dense\n",
        "\n",
        "model = Sequential()\n",
        "model.add(Embedding(max_words, embedding_dim, input_length=maxlen))\n",
        "model.add(Flatten())\n",
        "model.add(Dense(32, activation='relu'))\n",
        "model.add(Dense(1, activation='sigmoid'))\n",
        "model.summary()\n",
        "\n",
        "model.compile(optimizer='rmsprop',\n",
        "              loss='binary_crossentropy',\n",
        "              metrics=['acc'])\n",
        "history = model.fit(x_train, y_train,\n",
        "                    epochs=10,\n",
        "                    batch_size=32,\n",
        "                    validation_data=(x_val, y_val))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"sequential_1\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "embedding_1 (Embedding)      (None, 150, 100)          1000000   \n",
            "_________________________________________________________________\n",
            "flatten_1 (Flatten)          (None, 15000)             0         \n",
            "_________________________________________________________________\n",
            "dense_2 (Dense)              (None, 32)                480032    \n",
            "_________________________________________________________________\n",
            "dense_3 (Dense)              (None, 1)                 33        \n",
            "=================================================================\n",
            "Total params: 1,480,065\n",
            "Trainable params: 1,480,065\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "Epoch 1/10\n",
            "4/4 [==============================] - 2s 293ms/step - loss: 0.6990 - acc: 0.4752 - val_loss: 0.6924 - val_acc: 0.5119\n",
            "Epoch 2/10\n",
            "4/4 [==============================] - 1s 240ms/step - loss: 0.5086 - acc: 0.9661 - val_loss: 0.6925 - val_acc: 0.5160\n",
            "Epoch 3/10\n",
            "4/4 [==============================] - 1s 235ms/step - loss: 0.3354 - acc: 0.9693 - val_loss: 0.7030 - val_acc: 0.5106\n",
            "Epoch 4/10\n",
            "4/4 [==============================] - 1s 247ms/step - loss: 0.1941 - acc: 1.0000 - val_loss: 0.7128 - val_acc: 0.5052\n",
            "Epoch 5/10\n",
            "4/4 [==============================] - 1s 228ms/step - loss: 0.1177 - acc: 1.0000 - val_loss: 0.7054 - val_acc: 0.5092\n",
            "Epoch 6/10\n",
            "4/4 [==============================] - 1s 230ms/step - loss: 0.0648 - acc: 1.0000 - val_loss: 0.7018 - val_acc: 0.5237\n",
            "Epoch 7/10\n",
            "4/4 [==============================] - 1s 244ms/step - loss: 0.0453 - acc: 1.0000 - val_loss: 0.7043 - val_acc: 0.5179\n",
            "Epoch 8/10\n",
            "4/4 [==============================] - 1s 227ms/step - loss: 0.0255 - acc: 1.0000 - val_loss: 0.7107 - val_acc: 0.5238\n",
            "Epoch 9/10\n",
            "4/4 [==============================] - 1s 223ms/step - loss: 0.0186 - acc: 1.0000 - val_loss: 0.7180 - val_acc: 0.5217\n",
            "Epoch 10/10\n",
            "4/4 [==============================] - 1s 233ms/step - loss: 0.0139 - acc: 1.0000 - val_loss: 0.7202 - val_acc: 0.5252\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 545
        },
        "id": "8aoZM6k51Ote",
        "outputId": "3631e4cc-707e-4154-a446-279cf4dc9f19"
      },
      "source": [
        "acc = history.history['acc']\n",
        "val_acc = history.history['val_acc']\n",
        "loss = history.history['loss']\n",
        "val_loss = history.history['val_loss']\n",
        "\n",
        "epochs = range(1, len(acc) + 1)\n",
        "\n",
        "plt.plot(epochs, acc, 'bo', label='Training acc')\n",
        "plt.plot(epochs, val_acc, 'b', label='Validation acc')\n",
        "plt.title('Training and validation accuracy')\n",
        "plt.legend()\n",
        "\n",
        "plt.figure()\n",
        "\n",
        "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
        "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
        "plt.title('Training and validation loss')\n",
        "plt.legend()\n",
        "\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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3jyYNav1yxswYADLli0DYO7cufTu3Ztrrrmmfv6uXbvo0qVwPFVXV1NdXd3sOp588snWF9hOeQ/dzJq0cCHMmQNr1kBE8jxnTjK9lGbPns0ll1zC8ccfz7XXXstTTz3FCSecwLhx45g0aRIrV64E4PHHH+ess84Ckg+DCy+8kClTpnDUUUdxyy231C+vd+/e9e2nTJnCueeey8iRI5k1axZ1V1U/+OCDjBw5kgkTJnDllVfWLzfX6tWrOfHEExk/fjzjx49v8EHx3e9+l9GjRzNmzBiuuy65Ee2qVas49dRTGTNmDOPHj+eVV14p7YZqgvfQzaxJ118P27Y1nLZtWzJ9X/bSC6mtreXJJ5+kc+fObNmyhd/97nd06dKFRx55hK9//ev8/Oc/3+s1L730Eo899hhbt27l6KOP5tJLL93rUr9nn32W5cuXc/jhhzN58mR+//vfU11dzcUXX8wTTzzBsGHDmDlzZsGaBg8ezG9+8xt69OjByy+/zMyZM6mpqeGhhx7iF7/4BX/84x+pqKjgvffeA2DWrFlcd911nH322Wzfvp2PP/64tBupCQ50M2vS2rUtm74vzjvvPDp37gzA5s2bueCCC3j55ZeRxM6dOwu+5swzz6R79+50796dwYMH8/bbb1NZWdmgzcSJE+unjR07ltWrV9O7d2+OOuqo+ssCZ86cyYIFC/Za/s6dO7n88stZtmwZnTt35k9/+hMAjzzyCF/+8pepqKgAoH///mzdupV169Zx9tlnA8m15AeSu1zMrElHHtmy6fuiV69e9cPf/OY3Ofnkk3nhhRd44IEHGr2Er3v37vXDnTt3ZteuXa1q05ibbrqJQw45hOeee46amppmT9qWkwPdzJo0bx6kO6H1KiqS6fvT5s2bGTIkuVP3nXfeWfLlH3300bz66qusXr0agHvvvbfROg477DA6derE3Xffze7duwE47bTTuOOOO9iW9ke999579OnTh8rKShYvXgzAjh076ucfCA50M2vSrFmwYAEMHQpS8rxgQen7z/Nde+21fO1rX2PcuHEt2qMuVs+ePbntttuYOnUqEyZMoE+fPvTt23evdpdddhl33XUXY8aM4aWXXqo/ipg6dSrTpk2jurqasWPHMn/+fADuvvtubrnlFo499lgmTZrEW2+9VfLaG6Ny3UOruro6ampqyrJus47uxRdf5JOf/GS5yyi7999/n969exMRfOUrX2H48OFcffXV5S6rXqGfk6SnI6LgdZveQzezDuuHP/whY8eO5ZhjjmHz5s1cfPHF5S5pn/gqFzPrsK6++uo2tUe+r7yHbmaWEQ50M7OMcKCbmWWEA93MLCMc6GZ2wJ188sk8/PDDDabdfPPNXHrppY2+ZsqUKdRd6nzGGWewadOmvdrMnTu3/nrwxixevJgVK1bUj3/rW9/ikUceaUn5bZYD3cwOuJkzZ7Jo0aIG0xYtWtToDbLyPfjggxx88MGtWnd+oN94442ceuqprVpWW+NAN7MD7txzz+VXv/pV/X1RVq9ezRtvvMGJJ57IpZdeSnV1Nccccww33HBDwddXVVXx7rvvAjBv3jxGjBjBpz/96fpb7EJyjflxxx3HmDFj+OxnP8u2bdt48sknWbJkCV/96lcZO3Ysr7zyCrNnz+b+++8H4NFHH2XcuHGMHj2aCy+8kB07dtSv74YbbmD8+PGMHj2al156aa+a2sJtdn0dulkHd9VVkP6viZIZOxZuvrnx+f3792fixIk89NBDTJ8+nUWLFvG5z30OScybN4/+/fuze/duTjnlFJ5//nmOPfbYgst5+umnWbRoEcuWLWPXrl2MHz+eCRMmAHDOOedw0UUXAfCNb3yDH/3oR1xxxRVMmzaNs846i3PPPbfBsrZv387s2bN59NFHGTFiBF/60pf4wQ9+wFVXXQXAwIEDeeaZZ7jtttuYP38+t99+e4PXt4Xb7HoP3czKIrfbJbe75b777mP8+PGMGzeO5cuXN+geyfe73/2Os88+m4qKCg466CCmTZtWP++FF17gxBNPZPTo0SxcuJDly5c3Wc/KlSsZNmwYI0aMAOCCCy7giSeeqJ9/zjnnADBhwoT6G3rl2rlzJxdddBGjR4/mvPPOq6+72NvsVuTfAa0VvIdu1sE1tSe9P02fPp2rr76aZ555hm3btjFhwgRee+015s+fz9KlS+nXrx+zZ89u9La5zZk9ezaLFy9mzJgx3HnnnTz++OP7VG/dLXgbu/1u7m12P/744wN+L3TwHrqZlUnv3r05+eSTufDCC+v3zrds2UKvXr3o27cvb7/9Ng899FCTyzjppJNYvHgxH374IVu3buWBBx6on7d161YOO+wwdu7cycKc/5fXp08ftm7duteyjj76aFavXs2qVauA5K6Jn/nMZ4p+P23hNrsOdDMrm5kzZ/Lcc8/VB/qYMWMYN24cI0eO5Pzzz2fy5MlNvn78+PF8/vOfZ8yYMZx++ukcd9xx9fO+/e1vc/zxxzN58mRGjhxZP33GjBl873vfY9y4cQ1ORPbo0YM77riD8847j9GjR9OpUycuueSSot9LW7jNrm+fa9YB+fa57YNvn2tm1kE50M3MMqKoQJc0VdJKSaskXVdg/k2SlqWPP0na+zu5Zma2XzV72aKkzsCtwGlALbBU0pKIqL84NCKuzml/BTBuP9RqZiUUEUgqdxnWiNac3yxmD30isCoiXo2Ij4BFwPQm2s8EftbiSszsgOnRowcbNmxoVWjY/hcRbNiwocXXshfzxaIhwOs547XA8YUaShoKDAN+26IqzOyAqqyspLa2lvXr15e7FGtEjx49qKysbNFrSv1N0RnA/RGxu9BMSXOAOQBHHnlkiVdtZsXq2rUrw4YNK3cZVmLFdLmsA47IGa9MpxUygya6WyJiQURUR0T1oEGDiq/SzMyaVUygLwWGSxomqRtJaC/JbyRpJNAP+ENpSzQzs2I0G+gRsQu4HHgYeBG4LyKWS7pR0rScpjOAReGzLGZmZVFUH3pEPAg8mDftW3njc0tXlpmZtZS/KWpmlhEOdDOzjHCgm5llhAPdzCwjHOhmZhnhQDczywgHuplZRjjQzcwywoFuZpYRDnQzs4xwoJuZZYQD3cwsIxzoZmYZ4UA3M8sIB7qZWUY40M3MMsKBbmaWEQ50M7OMcKCbmWWEA93MLCMc6GZmGeFANzPLCAe6mVlGONDNzDLCgW5mlhEOdDOzjHCgm5llhAPdzCwjHOhmZhnhQDczy4iiAl3SVEkrJa2SdF0jbT4naYWk5ZJ+WtoyzcysOV2aayCpM3ArcBpQCyyVtCQiVuS0GQ58DZgcERslDd5fBZuZWWHF7KFPBFZFxKsR8RGwCJie1+Yi4NaI2AgQEe+UtkwzM2tOMYE+BHg9Z7w2nZZrBDBC0u8l/ZekqYUWJGmOpBpJNevXr29dxWZmVlCpTop2AYYDU4CZwA8lHZzfKCIWRER1RFQPGjSoRKs2MzMoLtDXAUfkjFem03LVAksiYmdEvAb8iSTgzczsACkm0JcCwyUNk9QNmAEsyWuzmGTvHEkDSbpgXi1hnWZm1oxmAz0idgGXAw8DLwL3RcRySTdKmpY2exjYIGkF8Bjw1YjYsL+KNjOzvSkiyrLi6urqqKmpKcu6zczaK0lPR0R1oXn+pqiZWUY40M3MMsKBbmaWEQ50M7OMcKCbmWWEA93MLCMc6GZmGeFANzPLCAe6mVlGONDNzDLCgW5mlhEOdDOzjHCgm5llhAPdzCwjHOhmZhnhQDczywgHuplZRjjQzcwywoFuZpYRDnQzs4xwoJuZZYQD3cwsI7qUuwAzs/YsArZvh23b9jw++KDwcN341Kkwfnzpa3Ggm+2DCHjrLVi+HFas2PO8di306QP9+hX36N8/ee7evdzvqGk7dsDGjXse771X3PDOndCtW/L+unXb+9HY9FK/ZufOxoO3mCBubF5Ey7Zj374OdLOyiYA339w7uFesSEKrTr9+cMwxcNJJyR/9xo1JuD/3XDK8dWvT6+nZs/gPgfxHjx7FvZddu2DTppYFct3wtm1NL/ugg/Z8ONVti379kjD96KPCjx07kuVu2lR4Xu747t3Fvcd9IUGvXlBRkTxyhwcNajieP7/Y8W7d9k/tDnSzHBHwxhsNQ7vuedOmPe3690/C6nOfS55HjUqeDzkkCYTGNBamjT1a8mHQo8feId+9+97r2rKl6eX06tXwyOETn9j7SKLQcN++0GU/J8ru3cledqGwb+xDIHd6167Nh2737k3/DNuyDhXoEfDhh/D++617bN3acDwCDj44efTtu2c4/1Fo3v76hLbi1AV3bmjXDW/evKfdgAFJUM+Y0TC4Bw9u3R99ly4wcGDyaKncD4NiHrW1SZD16weVlTB6dNOBXDfcln83O3dOHsUejXQ07S7QX34Zli0rHLDFPIrt65Kgd++9H4MGwbBhyTAkf/ybNsHbb8PKlcnwpk3NHxr27NnyD4HceT16tN+9iAMpAtatK9xVUii4zz+/YXAPGtR2tvO+fBhYx9DuAn3xYrj22obTunRJTkDlh29lZeFQLubRs2fr/5Ajkv7TTZv2BH7+I3/6hg3wyivJ9I0bk8PKpnTrVvyHQKHpFRVtJ6ha4+OPk37Xug/qDz5InjdtSj5Yc8M7t4th0KAkrGfN2hPao0Yle9xm7Z2ipadnS6S6ujpqampa/Lp33oF3320Yvm35ELE16i6Dau5DoKnp27c3vY4uXVr+IZA7vU+f4j4Qcru56kI3P4Sbm1dovLmTc4MGJWGdu7c9alQy3aw9k/R0RFQXmlfUHrqkqcA/A52B2yPiO3nzZwPfA9alk74fEbe3uuImDB6c/b0pKTlC6NkTDjusdcvYsaPxD4DGPgTefHPPvA8+aHr5nTo1DP2+fZM+3kIh3JJ9hp49G35Y9+qVPB9ySMPx/OG68T59kpN4Dm7riJoNdEmdgVuB04BaYKmkJRGxIq/pvRFx+X6o0Vqhe/d9+/DbuXPv0G/uA6JbNzjyyMJB21QI1w1XVCQnvMysdYrZQ58IrIqIVwEkLQKmA/mBbhnStatPwJm1N8Xcy2UI8HrOeG06Ld9nJT0v6X5JRxRakKQ5kmok1axfv74V5ZqZWWNKdXOuB4CqiDgW+A1wV6FGEbEgIqojonpQKzo5Fy6Eqqqk/7aqKhk3M7NEMYG+Dsjd465kz8lPACJiQ0TsSEdvByaUprw9Fi6EOXNgzZrkJNuaNcm4Q93MLFFMoC8FhksaJqkbMANYkttAUu61GNOAF0tXYuL66/e+VG3btmS6mZkVcVI0InZJuhx4mOSyxR9HxHJJNwI1EbEEuFLSNGAX8B4wu9SFrl3bsulmZh1Nu/liUVVV0s2Sb+hQWL26ZGWZmbVpTX2xqN38x6J585LrlHNVVCTTzcysHQX6rFmwYEGyRy4lzwsWJNPNzKyd3Zxr1iwHuJlZY9rNHrqZmTXNgW5mlhEOdDOzjHCgm5llhAPdzCwjyvbFIknrgQJfFWpXBgLvlruINsTbYw9vi4a8PRral+0xNCIK3t2wbIGeBZJqGvvGVkfk7bGHt0VD3h4N7a/t4S4XM7OMcKCbmWWEA33fLCh3AW2Mt8ce3hYNeXs0tF+2h/vQzcwywnvoZmYZ4UA3M8sIB3orSDpC0mOSVkhaLulvyl1TuUnqLOlZSb8sdy3lJulgSfdLeknSi5JOKHdN5STp6vTv5AVJP5PUo9w1HSiSfizpHUkv5EzrL+k3kl5On/uVan0O9NbZBfxtRIwCPgV8RdKoMtdUbn/Dfvhfsu3UPwO/joiRwBg68HaRNAS4EqiOiP9B8m8sZ5S3qgPqTmBq3rTrgEcjYjjwaDpeEg70VoiINyPimXR4K8kf7JDyVlU+kiqBM4Hby11LuUnqC5wE/AggIj6KiE3lrarsugA9JXUBKoA3ylzPARMRT5D8n+Vc04G70uG7gL8s1foc6PtIUhUwDvhjeSspq5uBa4GPy11IGzAMWA/ckXZB3S6pV7mLKpeIWAfMB9YCbwKbI+I/yltV2R0SEW+mw28Bh5RqwQ70fSCpN/Bz4KqI2FLuespB0lnAOxHxdLlraSO6AOOBH0TEOOADSnhI3d6k/cPTST7oDgd6SfpCeatqOyK5brxk14470FtJUleSMF8YEf9e7nrKaDIwTdJqYBHw55LuKW9JZVUL1EZE3RHb/SQB31GdCrwWEesjYifw78CkMtdUbm9LOgwgfX6nVAt2oLeCJJH0kb4YEf9U7nrKKSK+FhGVEVFFcrLrtxHRYffAIuIt4HVJR6eTTgFWlLGkclsLfEpSRfp3cwod+KGeVaYAAACISURBVCRxaglwQTp8AfCLUi3Ygd46k4EvkuyNLksfZ5S7KGszrgAWSnoeGAv8Q5nrKZv0SOV+4Bngv0kyp8PcBkDSz4A/AEdLqpX0V8B3gNMkvUxyBPOdkq3PX/03M8sG76GbmWWEA93MLCMc6GZmGeFANzPLCAe6mVlGONDNzDLCgW5mlhH/H/kLmHjgPkUmAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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vHfvXgYNCk7D/+1vf5vy5Q8fPpxt27axfft2AJYuXdroa8466yyKw+RYtWoVAwYM4KijjuKtt97i5JNP5nvf+x7jxo1jy5Yt7Nixg6OPPprvfOc7XHXVVWzYsCHlP0NHVXOgfscOcD98oF6hLunSbgMdgvDevh0+/zx4TMc5yLfeeivf//73GTNmTMr3qAF69OjBr371KyZPnszYsWPp3bs3ffr0afA1CxYsYP369YwaNYp58+bx8MMPA3Dvvfdy0kknMWrUKLp06cKUKVNYtWoVp5xyCmPGjGHp0qXccMMNKf8ZOipdIEwyzWrOrEi3goICLykpqdP2+uuvc+KJJ2aknrZk//799OrVC3fn2muvZdiwYdx0002ZLusI+n3V1alTsGcezyzY6RBJBTNb7+4Jz5Fu13voUXX//fczevRovvzlL7N3716uvvrqTJckSajvgLy+FC3p0m7Pcomym266qU3ukUvDFi2q+2U30AXCJL20hy6SIm3pAmE626Zj0h66SAq1hQuE6XZ8HZf20EUiRmfbdFwKdJGI0e34Oi4FeoxzzjmHFStW1Gm79957mTt3br2vmTRpEjWnX15wwQXs2bPniD4LFizgnnvuaXDdTz31FJs3H76r3+23386zzz7blPIT0mV2Ox6dbdNxKdBjzJw5kyVLltRpW7JkSVIXyILgKol9+/Zt1rrjA33hwoWcd955zVqWdGxt6XZ8OjibXm32oOiNN0J4afKUGT0a7r23/uenT5/ObbfdxqFDh+jatSvbt2/nvffe46yzzmLu3LmsW7eOAwcOMH36dH70ox8d8fr8/HxKSkoYMGAAixYt4uGHH2bgwIEMHjyYsWPHAsE55kVFRRw6dIjjjz+eRx55hNLSUpYtW8bq1av58Y9/zBNPPMGdd97JRRddxPTp01m5ciW33HIL1dXVjBs3jvvuu49u3bqRn5/PlVdeydNPP01VVRV/+MMfGDFiRL0/38cff8zs2bPZtm0b2dnZFBUVMWrUKFavXl37jVEzY82aNezfv5/LLruMffv2UV1dzX333cdZZ53Vsl+ApEXNgc/584Nhlry8IMzTfUBUB2fTT3voMfr168f48eN55plngGDv/Jvf/CZmxqJFiygpKWHjxo2sXr269prjiaxfv54lS5ZQWlrK8uXLWbduXe1zl1xyCevWreOVV17hxBNP5MEH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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "L0z_tlbgFwtC"
      },
      "source": [
        "Evaluate the model on the test set"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MGiarT-cF9G6"
      },
      "source": [
        "Tokenize the data"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eOcFf61ogwsi"
      },
      "source": [
        "test_dir = os.path.join(imdb_dir, 'test')\n",
        "\n",
        "labels = []\n",
        "texts = []\n",
        "\n",
        "for label_type in ['neg', 'pos']:\n",
        "    dir_name = os.path.join(test_dir, label_type)\n",
        "    for fname in sorted(os.listdir(dir_name)):\n",
        "        if fname[-4:] == '.txt':\n",
        "            f = open(os.path.join(dir_name, fname))\n",
        "            texts.append(f.read())\n",
        "            f.close()\n",
        "            if label_type == 'neg':\n",
        "                labels.append(0)\n",
        "            else:\n",
        "                labels.append(1)\n",
        "\n",
        "sequences = tokenizer.texts_to_sequences(texts)\n",
        "x_test = pad_sequences(sequences, maxlen=maxlen)\n",
        "y_test = np.asarray(labels)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8QZFDXXyGGgH"
      },
      "source": [
        "Evaluate the model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6tDQ81lADEEH",
        "outputId": "7c440c4f-d628-4f9a-f9c7-3a83c677e542"
      },
      "source": [
        "model.load_weights('pre_trained_glove_model.h5')\n",
        "model.evaluate(x_test, y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "782/782 [==============================] - 2s 3ms/step - loss: 0.7657 - acc: 0.5491\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[0.7656670808792114, 0.5491200089454651]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bqm3PA7WGO9U"
      },
      "source": [
        "validation accuracy is 54%"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Pretrained word embedding has a validation accuracy of 56%, while an embedding layer has a validation accuracy of 54%. Pretrained word embedding provided us with a little higher accuracy.\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "_-QqSIeLklBG"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}