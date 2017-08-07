---
layout: post
title: "Learning Math with LSTMs and Keras"
description: "Let's build a neural network that can do math. In Keras."
categories: machine-learning
permalink: learning-math/
tags: [machine-learning, recurrent-neural-networks, keras]
disqus: true
---

Since ancient times, it has been known that machines excel at math while humans
are pretty good at detecting cats in pictures. But with the advent of deep
learning, things have started to make less sense...

Today we want to teach the machines to do math... Again! But instead of feeding
them with optimized, known representations of numbers and calling hard-coded
operations on them, we will feed in math formulas and their results, character
by character, and have the machine figure out how to interpret them.

The result is much more human: Not all results are exact, but they are close.
It's more of an approximation than an exact calculation. As a human, I surely
can relate to almost-correct math, if anything.

Today's post will be much more practical than usual, so if you want to build
this at home, get your Python ready and your virtualenv up to speed! You can
find the complete code with some changes [here](https://github.com/cpury/lstm-math).


### Recurrent Neural Networks and Long-Short-Term-Memory

As in my previous post, we're going to use Recurrent Neural Networks, or RNN
for short. More specifically, a kind of RNN known under the fancy name of
Long-Short-Term-Memory, or LSTM. If you need a primer on these, I can recommend
[Christopher Olah's "Understanding LSTMs"](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
and again
[Andrej Karpathy's "The Unreasonable Effectiveness of Recurrent Neural Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).


### Sequence to Sequence Learning

Sequence to sequence learning deals with problems where a source sequence of
inputs has to be mapped to a target sequence of outputs where each output
is not necessarily directly depending on a single input. The classical
example is translation. How do you learn that a Chinese input phrase
"他现在已经在路上了。" equals "She is on her way." in English?

Even with RNNs it's not directly obvious how to do this. Well, it wasn't until
Google published their paper
["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215)
(Seq2Seq) in 2014. The idea is to train a joint encoder-decoder model.
First, an encoder based on RNNs learns an abstract, (ideally)
language-independent representation. Then a decoder also based on RNNs learns
to decode it to another language, generating a new sequence from the encoding
as output.


### The setup

Data is usually hard to come by, labeled data even more so. But math equations
are cheap: They can be easily generated, and Python gives us their result
simply by calling `eval(equation_string)`.
Starting with simple addition of small, natural numbers, we can easily generate
a lot of equations along with their results and train a Seq2Seq model on it.

For example, a typical datapoint for addition could look like this:
```
input = '31 + 87'
output = '118'
```

Since we're learning on just a fraction of all possible formulas, the model
can't just learn the results by heart. Instead, in order to generalize
to all other equations, it really needs to "understand" what addition "means".
This would include, among others:
* An equation consists of numbers and operations
* Two numbers are added up digit-by-digit
* If two digits add up to a value higher than 9, they carry over to the next
* Commutative axiom: `a + b = b + a`
* etc.

And this just for additions! If we add more operations, or mix them, the
network needs to grog even more rules like this.


### Generating math formulas as strings

For now, let's build some equations! We're going to go real simple here and
just work with easy-peasy addition of two small natural numbers. Once we get
that to work, we can build more funky stuff later on.

To make sure we build each possible equation only once, we're going to use a
generator. With the mighty `itertools` Python standard library, we can generate
all possible permutations of two numbers and then generate a formula for each.
Here's our basic code. Note that I'm referencing some global variables (IN
ALL CAPS), we will define them later.

{% highlight python %}
import itertools
import random

def generate_all_equations(shuffle=True, max_count=None):
    """
    Generates all possible math equations given the global configuration.
    """
    # Generate all possible unique sets of numbers
    number_permutations = itertools.permutations(
        range(MIN_NUMBER, MAX_NUMBER + 1), 2
    )

    # Shuffle if required. The downside is we need to convert to list first
    if shuffle:
        number_permutations = list(number_permutations)
        random.shuffle(number_permutations)

    # If a max_count is given, use itertools to only look at that many items
    if max_count is not None:
        number_permutations = itertools.islice(number_permutations, max_count)

    # Build an equation string for each and yield to caller
    for x, y in number_permutations:
        yield '{} + {}'.format(x, y)
{% endhighlight %}

For our first experiment, let's add this global config:

{% highlight python %}
MIN_NUMBER = 0
MAX_NUMBER = 999
{% endhighlight %}

This will generate us equations like this:

```
'89 + 7'
'316 + 189'
'240 + 935'
```

For any of these equation strings, we can easily get the result in Python using
`eval(equation)`.

That was easy, right?

### Encoding strings for Neural Networks

Remember we want to look at the strings as sequences. The RNN will not see the
input string as a whole, but one character after the other. It will then have
to learn to remember the important parts in its internal state. That means we
need to convert each input and output string into vectors first.

We do this by encoding each character as a one-hot vector. Each string is then
simply a matrix of these character-vectors. The way we encode is quite
arbitrary, as long as we decode it the same way later on. Here's some
code I used for this:

{% highlight python %}
import numpy as np

def one_hot(index, length):
    """
    Generates a one-hot vector of the given length that's 1.0 at the given
    index.
    """
    assert index < length

    array = np.zeros(length)
    array[index] = 1.

    return array


def char_to_one_hot_index(char):
    """
    Given a char, encodes it as an integer to be used in a one-hot vector.
    Will only work with digits, dots and +-signs, everything else
    (including spaces) will be mapped to a single value.
    """
    if char.isdigit():
        return int(char)
    elif char == '+':
        return 10
    elif char == '.':
        return 11
    else:
        return 12


def char_to_one_hot(char):
    """
    Given a char, encodes it as a one-hot vector based on the encoding above.
    """
    return one_hot(char_to_one_hot_index(char), 13)


def one_hot_index_to_char(index):
    """
    Given an index, returns the character encoded with that index.
    Will only work with encoded digits, dots or +, everything else will
    return the space character.
    """
    if index <= 9:
        return str(index)

    if index == 10:
        return '+'

    if index == 11:
        return '.'

    return ' '


def one_hot_to_char(vector):
    """
    Given a one-hot vector, returns the encoded char.
    """
    indices = np.nonzero(vector == 1.)

    assert len(indices) == 1
    assert len(indices[0]) == 1

    return one_hot_index_to_char(indices[0][0])


def one_hot_to_string(matrix):
    """
    Given a matrix of single one-hot encoded char vectors, returns the
    encoded string.
    """
    return ''.join(one_hot_to_char(vector) for vector in matrix)
{% endhighlight %}


### Building the model in Keras


### Training


### Results


### Further Experiments
