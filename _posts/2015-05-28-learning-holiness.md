---
layout: post
title: "Learning Holiness"
date:   2015-05-28 18:33:45
description: "Learning how to produce Bible-like texts with Recurrent Neural Networks"
categories: machine-learning
permalink: learning-holiness/
tags: [machine-learning, recurrent-neural-networks]
disqus: true
---

Let's teach a neural network to write Bible-like texts! If you have not heard about [Andrej Karpathy](http://karpathy.github.io/)'s incredibly awesome [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) about Recurrent Neural Networks, I recommend you drop everything right now and head over there for a great read! All of this here is based on his work and provided code.

##### TL;DR, technical bits
Recurrent Neural Networks are a variant of Machine Learning algorithms that work with sequential data as inputs and/or outputs, e.g. text data. Here we let them look at large bodies of text, and based on one part of the text teach it to predict the next. You can then sample this prediction to produce text the network thinks to be likely. Since this is a stochastic process, you can also tell it how "inventive" it should be while doing this.

##### TL;DR, fun bits
This sampled data can look a lot like the data you learned from, e.g. Andrej lets it create fake Shakespeare prose, fake Wikipedia articles, fake math papers, etc. The results are surprisingly smart in some ways, but also hilariously dumb in others. Imagine all the fun we can have with this! One drawback is that you need huge amounts of data to train - in a way the model needs to learn the English language from scratch.

### Idea
I instantly went to work with what any sane person would have done: I started amassing copious amounts of erotic Harry Potter fanfiction, which - as is common knowledge - amounts to about 20% of the Internet's plain text data. ([^1]) But this endeavor got boring quickly... After one hour of skimming terribly written, dirty dirty paragraphs and copy-pasting them to a text file, I only had little more than one megabyte of data. So I went for the second best option: The Bible!

### Experiment
The King James version of the Bible is well-known and easily accessible, while also amounting to a beautiful 4.4 MB of data. How could I resist the temptation?

As for the settings, I was lazy and also bound by my CPU, so I went for Andrej's default parameters.

### Results

> 26:24 And the children of Israel went up from the LORD the word of the LORD, and set the
> LORD in the house of the LORD.

Teehee. You might wonder what infinite loop I get stuck in setting the sampling to a low temperature (aka what's the most cliché sequence in the Bible?). Well, remember those family trees in the Old Testament?

> 1:12 And the sons of Jehoiada, and Jehoshaphat the son of Jehoiada, and
> Jehoshaphat the son of Jehoiada, and Jehoshaphat the son of
> Joseph, and the sons of Jehoiadan the son of Jehoiada, and
> Jehoshaphat the son of Jehoiada, and the sons of Jehoiadan
> the son of Jehoiada, and Jehoshaphat the son of Jehoiadan the son of
> Joseph the son of Jehoiada, and Jehoshaphat the son of Jehoiada, and
> Jehoshaphat the son of Jehoiada.

Just like in *One Hundred Years of Solitude*!

What about setting the temperature high, i.e. letting it become more experimental?

> 8:13 And Moses, leported is with all men, so Job, thereof: herings into
> the temple of Beliah.

BELIAH DEMANDS HERRINGS!

Here are some more of my favorites at various temperatures:

> 4:24 And the LORD said unto him, Behold, I have sent the burnt offering, and the dust of the sea, and delivered
> against the LORD the priests and the head of the house of Moses.

Dramatic!

> 24:10 And the LORD said unto the LORD, Behold, I will set thee out of the
> children of Israel, and the LORD shall be a son of the LORD.

Haha, that would be hilarious! Oh, wait.

> 2:1 There is no father's house.

That's pretty deep, man...

> 1:12 And there was not a little care any bear, and they thereof, and the
> wicked of the LORD shall be the father of the house of his heart.

Care Bears..?

> 1:15 And the sons of Aaron, and the sons of Jesse said, I am the LORD.

Are you, now?

> 19:20 And the LORD said unto Moses, God is the streets of the sons of
> Mattan, the son of Jeroboam the son of Ziphath.

God speaks in metaphors.


### Analysis

I am really happy about the results. I literally downloaded the code, grabbed a plain text copy of the Bible and started training a minute later. The trained model seems to understand paragraphs and has some basic notions about sentence structure and the functions of words.

Still, even though the output shown here seems *slightly* coherent, I can promise you that most of it is not. It's a little disappointing that while we get correct paragraphs with line numbers, the line numbers do not follow any order. Also, their length vary greatly.

Overall, I assume the 4+ million characters of the text are not enough to get overly confident about the concepts presented. Maybe I should add other translations? Other holy texts?

I wonder if we can take advantage of Transfer Learning here, i.e. have a basic model run on as much English text of all epochs and styles as possible, and then seed the sampling with some Bible quote?

Anyway, please leave your comments and own experiments or ideas in the comment section below!

**Amen!**


### Disclaimer

I'm an absolute layperson when it comes to the Bible / Christianity / being religious. My main reasons are earnestly to have a fun little experiment with an amazing new Machine Learning technique and a well-known, just-go-for-it dataset. If you feel like this insulted you or your religion, I'm terribly sorry about that.


[^1]: **Source:** Common knowledge. I told you already.
