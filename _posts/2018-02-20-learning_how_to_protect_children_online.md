---
layout: post
title: "Learning how to Protect Children Online"
date: 2018-06-20 14:25:51
description: "How I built a neural network that detects perverts and other dangers"
categories: machine-learning
permalink: learning-to-protect/
tags: [machine-learning]
disqus: true
---
Today I'm not going to talk about a funny use of Machine Learning or a cool application you can try at home. Instead, I want to talk about a project I’ve been working on professionally over the last few months. The job involved a **complex NLP pipeline**, an **interesting application**, and lots of new experiences for me. The knowledge I gathered helped me get in the **[top 4%](https://www.kaggle.com/mschumacher) of the [Google Jigsaw Toxic Comments on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**, and maybe it can also help you with your projects in the future.


### Privalino
**Privalino** is a young startup from Germany that aims to make the internet a **safer place for kids**. It's first project is to build an **instant messenger** for children under the age of 13 that can **automatically detect dangers** such as cyber grooming, cyber bullying and others so that appropriate measures can be taken. I've joined their small team a while ago and am quite excited about the project! If you are interested, you can find more info in German at [privalino.de](https://privalino.de/).


### The Problem: Cyber Grooming
You might not be aware of this, but these days, a lot of really young kids have smartphones in their pockets. We all know that the internet is a really messed up place. But what's often even more dangerous is strangers with bad intentions. Even the smartest kids are easy to manipulate if you push the right buttons, and in most cases they don't realize what's happening, or that they should get help. I'm not generally for surveilling or controlling kids' behavior, but if we can save even a single young kid from a pervert without messing up their childhood in some other way, I think we are doing something good.

So let's look at this problem from the perspective of AI. An algorithm, much like a human, would look at the last X messages in a conversation and then try to induce whether something fishy is going on. Cyber grooming can look any number of ways. Maybe a stranger is trying to arrange a meeting in person. Maybe they are trying to get naked pictures. Maybe they are trying to initiate sexy talk. Or maybe they are trying to get them to change apps so they can talk in private.

The problem is even more subtle than that, however. The chat partner might be a real classmate wanting to meet up after school. Parents might send their kids photos of their dogs. Maybe two children are innocently using naughty language that they overheard before. And to make it even more complicated, maybe the person they think to be their crush from school is actually someone else, pretending.

Any way you turn it, it's a **very complex problem**. A simple binary classification as in "grooming" vs "no grooming" is not enough. Even humans looking at the chat can not always know for sure. Instead, we treat it as a **regression** problem to predict a **subjective likelihood** of grooming happening. Essentially, we ask the annotators this: After reading through these messages, based on your gut feeling, how likely do you think this to be a dangerous conversation?


### Data

So you might be wondering: Where the hell did you get the data to learn this? Well, there's an abundance of "normal" chat conversation datasets, but to get "dangerous" examples, we had to enter... let's call it an "ethical grey area". What we did was this: We entered public chat rooms for kids, posing as young children ourselves, waiting for creeps to chat us up. We thought we might get a few data points that way. How wrong we were... We got **tens of thousands** with almost no effort!

Afterwards, we went through the data by hand and **annotated** it. Each message got a score between 0 and 1, representing our perceived chance of danger in the conversation up to this message. Most conversations begin at 0.5 (unsure if danger is present or not). If it becomes obvious that it's a friend from school or a parent, the danger might slowly go down. If after two messages someone inquires about the color of the other's undergarments, we might increase the danger value.


### Proposal

Having collected a suitable dataset, we aimed to build a model that could predict this score without overfitting to the training set. The Privalino team had already trained a traditional NLP classifier that delivered a solid performance. However, these methods are not suitable for solving such an intricate problem
to full satisfaction.

Instead, I built a model based on **Deep Learning ideas**. A very similar architecture also helped me get in the [top 4%](https://www.kaggle.com/mschumacher) of the [Google Jigsaw Toxic Comments competition on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). I hope this will help you with your practice as well. The final model achieved much better metrics compared to the traditional approach. And it only takes a fraction of the time to run the prediction.


### The Model

#### Input

To handle typos, extra spaces, etc., I first considered a **character-level encoding**, where the input is a sequence of one-hot vectors representing single characters. This promises the strongest robustness against small variations such as typos. We can't leverage unspervised pretraining though, so the model has to learn all words by itself.

I also tried a word-based model with pretrained **FastText embeddings**. Unlike previous word embedding algorithms, it can deal with typos and unseen words quite well. By training it with no supervision on a large text corpus, we can already pre-learn a lot of concepts and nuances before we even look at our limited labeled dataset.

Both character-level and word-level encodings have their strengths and weaknesses and shift the complexity of the problem a bit. In the future, I want to explore approaches combining the two. For the moment, I got my best results using FastText embeddings. The FT model was trained on a large corpus of Twitter data and all German text datasets I could get my hands on. The resulting word vector space is 200-dimensional.

For practical reasons, we have to limit the **window** of text the model can look at, at least during training. Whenever I found a good architecture on smaller windows (~100 words), I fine-tuned it afterwards on much larger windows (~1000 words). Messages were separated by a special token, and I experimented with different ways to encode who is speaking at the moment, but the differences in accuracy was small.

#### Architecture

When working with sequences in neural networks, you have two basic building blocks: **1D-Convolutions** and **recurrent neural networks**. Convolutions are great at learning local features, while RNNs excel at learning long-term dependencies. I tried convolutions alone (with a global max pooling layer at the end), but their power of understanding a text just seemed sub-par to LSTMs. I also tried pure RNN models, but they are cumbersome to train and need more resources. So my final architecture involved both: A conv block followed by an RNN block.

##### Conv Block

Given a raw text, we split it into words and retrieve the embedding vector of each. This results in a sequence with 200 features. Next, we apply a **one-dimensional convolution looking at three words each**. They generate 100 location-independent, non-linear features. Finally, to reduce the length
of the sequence, we apply a **max pooling over each 3 elements**, capturing only the strongest signals in each location. So at the end of the conv block, we have both reduced the number of features as well the length of the sequence.

{% include svgs/privalino_cnn.svg %}

##### RNN Block

Now we apply **75 units of Long-Short-Term-Memory (LSTM)** to this compressed sequence. Each step carries forward the state and an output signal. This learns complex long-term relations in the sequence. Finally, we get a **final state** that we can use as output of this layer.

{% include svgs/privalino_rnn_1.svg %}

This makes learning **hard**, though: Let's say the RNN discovers an important feature right in the beginning of the sequence. To pass this signal to the next layer is not easy, because it needs to be passed through all the following time steps until it reaches the final state.

This could be solved by **attention**, but my experiments using it all led to a strong overfit.

Instead I opted for a **much simpler solution**: We can take each time step's output and aggregate them globally. E.g. applying a **global max pooling** to all the outputs lets a feature be useful right away simply by increasing its value. **Global average pooling** can help get a general feel of the whole text.

Both these two pooling operations are applied to the LSTM outputs and concatenated to the final state. That way, the output of the RNN block is in the shape of 3x75 fixed features.

{% include svgs/privalino_rnn_2.svg %}

##### Finalizing

Next, I apply a simple sigmoid output layer to these features to get the model's output. Additionally, I apply **batch normalization** and **(structural) dropout** wherever helpful. The structual part of the dropout is important: In sequences, it is important to delete whole features instead of single values.


### Training

I found **Nadam (Adam with Nesterov-Momentum)** to be the best optimizer for RNN models. **Gradient clipping** is important to keep the training stable. The **learning rate** was set as high as possible. With this, training took something in the order of 30 minutes.


### Results

To evaluate the model, I had spared out 20% of our training conversations for the test set. On those, the model achieved a **Pearson-correlation of 0.96** between the predicted scores and our human labels. **Errors larger than 0.4** were only made extremely rarely in about **0.1%** of all cases. Running a single prediction takes **10 ms** on a basic server without GPU.


### Ethics

So we've proven that the problem is technically **solvable** to an acceptable degree. But **should it even be solved?**

We are well aware that this whole project can make some people cringe, and that's good. We are spying on kids and censoring them, in a way. However, I think in the case of really young kids, the danger that we're protecting them from is real. And even with the healthy attitude that children should be allowed to make their own mistakes, we should agree that no child deserves to be abused. At the end of the day, our service is a **tool for the parents**. They should wield it responsibly. To this end, Privalino tries their best to **integrate and talk to the children and parents directly**.


### Outlook

Needless to say, there is still room for improvement. We are constantly collecting more data, and there are still many more complex and subtle tactics that we are not catching right now. I'm always happy to hear about your ideas.
