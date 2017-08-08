---
layout: post
title: "Learning Math with LSTMs and Keras"
date:   2015-08-08 15:58:50
description: "Let's build a neural network that can do math. In Keras."
categories: machine-learning
permalink: learning-math/
tags: [machine-learning, recurrent-neural-networks, keras]
disqus: true
draft: true
---

Since ancient times, it has been known that machines excel at math while humans
are pretty good at detecting cats in pictures. But with the advent of deep
learning, things have started to make less sense...

Today we want to teach the machines to do math... Again! But instead of feeding
them with optimized, known representations of numbers and calling hard-coded
operations on them, we will feed in math formulas along with their results,
character by character, and have the machine figure out how to interpret them
and arrive at the results by itself.

The trained model is much more human: Not all results are exact, but they are
close. It's more of an approximation than an exact calculation. As a human, I
surely can relate to almost-correct math, if anything.

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
* Commutativity: `a + b = b + a`
* Adding zero does not actually do anything: `a + 0 = a`
* etc.

And this just for additions! If we add more operations, or mix them, the
network needs to grog even more rules like this.

### Setting up your environment

If you're doing this from scratch, you will want to work in a new virtualenv.
I also recommend using Python 3, because it's really time we got over Python
2.x... If you don't know how virtualenv works, I really recommend looking into
them. You can find a short tutorial
(here)[http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/].

Once your virtualenv is up and running, install these requirements:

```
pip install numpy tensorflow keras
```

If you run into trouble with Tensorflow, follow
(their guide)[https://www.tensorflow.org/install/].

Great! You can paste the code below into a shell, or store it in a file, it's
up to you.


### Generating math formulas as strings

For now, let's build some equations! We're going to go real simple here and
just work with easy-peasy addition of two small natural numbers. Once we get
that to work, we can build more funky stuff later on. See the
(full code)[https://github.com/cpury/lstm-math] to find out how to build more
complex equations.

To make sure we build each possible equation only once, we're going to use a
generator. With the handy `itertools` Python standard package, we can generate
all possible permutations of two numbers and then create a formula for each.
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

Assuming these global config values ...

{% highlight python %}
MIN_NUMBER = 0
MAX_NUMBER = 999
{% endhighlight %}

... will generate us equations like this:

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
    else:
        return 11


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

    return ' '


def one_hot_to_char(vector):
    """
    Given a one-hot vector, returns the encoded char.
    """
    indices = np.nonzero(vector == 1.)

    assert len(indices) == 1
    if not len(indices[0]) == 1:
        return ''

    return one_hot_index_to_char(indices[0][0])


def one_hot_to_string(matrix):
    """
    Given a matrix of single one-hot encoded char vectors, returns the
    encoded string.
    """
    return ''.join(one_hot_to_char(vector) for vector in matrix)
{% endhighlight %}

With these helper functions, it is quite easy to generate the dataset:

{% highlight python %}
def build_dataset():
    """
    Builds a dataset based on the global config.
    Returns (x_test, y_test, x_train, y_train).
    """
    generator = generate_all_equations(max_count=N_EXAMPLES)

    # Split into training and test set based on SPLIT:
    n_test = round(SPLIT * N_EXAMPLES)
    n_train = N_EXAMPLES - n_test

    x_test = np.zeros(
        (n_test, MAX_EQUATION_LENGTH, N_FEATURES), dtype=np.bool
    )
    y_test = np.zeros(
        (n_test, MAX_RESULT_LENGTH, N_FEATURES), dtype=np.bool
    )

    # Get the first n_test equations and convert to test vectors
    for i, equation in enumerate(itertools.islice(generator, n_test)):
        result = str(eval(equation))
        # Pad the result with spaces
        result = ' ' * (MAX_RESULT_LENGTH - len(result)) + result

        for t, char in enumerate(equation):
            x_test[i, t, char_to_one_hot_index(char)] = 1

        for t, char in enumerate(result):
            y_test[i, t, char_to_one_hot_index(char)] = 1

    x_train = np.zeros(
        (n_train, MAX_EQUATION_LENGTH, N_FEATURES), dtype=np.bool
    )
    y_train = np.zeros(
        (n_train, MAX_RESULT_LENGTH, N_FEATURES), dtype=np.bool
    )

    # The rest will go to the train set
    for i, equation in enumerate(generator):
        result = str(eval(equation))
        result = ' ' * (MAX_RESULT_LENGTH - len(result)) + result

        for t, char in enumerate(equation):
            x_train[i, t, char_to_one_hot_index(char)] = 1

        for t, char in enumerate(result):
            y_train[i, t, char_to_one_hot_index(char)] = 1

    return x_test, y_test, x_train, y_train
{% endhighlight %}

And later to print out some examples along with their targets:

{% highlight python %}
from __future__ import print_function

def prediction_to_string(matrix):
    """
    Given the output matrix of the neural network, takes the most likely char
    predicted at each point and returns the whole string.
    """
    return ''.join(
        one_hot_index_to_char(np.argmax(vector))
        for vector in matrix
    )

def print_example_predictions(count, model, x_test, y_test):
    """
    Print some example predictions along with their target from the test set.
    """
    print('Examples:')

    # Pick some random indices from the test set
    prediction_indices = np.random.choice(
        x_test.shape[0], size=count, replace=False
    )
    # Get a prediction of each
    predictions = model.predict(x_test[prediction_indices, :])

    for i in range(count):
        print('{} = {}   (expected: {})'.format(
            one_hot_to_string(x_test[prediction_indices[i]]),
            prediction_to_string(predictions[i]),
            one_hot_to_string(y_test[prediction_indices[i]]),
        ))
{% endhighlight %}


### Building the model in Keras

Now, let's build the model. Thanks to Keras, this is quite straightforward.

First, we need to decide what the shape of our inputs is supposed to be. Since
it's a matrix with a one-hot-vector for each position in the equation string,
this is simply `(MAX_EQUATION_LENGTH, N_FEATURES)`. We'll pass that input to a
first layer consisting of 128 LSTM cells. Each will look at the input,
character by character, and output a single value.

Now, ideally we'd use the activation of each LSTM for each of the sequence
items and then use them as outputs to our decoder. It seems Keras is having
a hard time with this, so the accepted workaround is to let the LSTM cells
output single values each, and then repeat these vectors using `RepeatVector`
to feed the decoder network. There's an issue on Keras discussing this
[here](https://github.com/fchollet/keras/issues/5203).

So after we repeat the encoded vector `n` times with `n` being the (maximum)
length of our output sequences, we run this repeat-sequence through the
decoder: An LSTM layer that will output sequences of vectors. Finally, we want
to combine each LSTM cell's output at each point in the sequence to a single
output vector. This is done using Keras' `TimeDistributed` wrapper around a
simple `Dense` layer.

Since we expect something like a one-hot vector for each output character,
we still need to apply `softmax` as usual in classification problems.

Note that from what I gathered, this way of building Seq2Seq models in Keras
is not optimal and not exactly equivalent to what is proposed in the paper. It
works nonetheless. For a more correct implementation, try out
[Fariz Rahman's seq2seq package](https://github.com/farizrahman4u/seq2seq).

Either way, here is our code:

{% highlight python %}
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, Dense, Activation
from keras.layers.wrappers import TimeDistributed

def build_model():
    """
    Builds and returns the model based on the global config.
    """
    input_shape = (MAX_EQUATION_LENGTH, N_FEATURES)

    model = Sequential()

    # Encoder:
    model.add(LSTM(128, input_shape=input_shape))

    # The RepeatVector-layer repeats the input n times
    model.add(RepeatVector(MAX_RESULT_LENGTH))

    # Decoder:
    model.add(LSTM(128, return_sequences=True))

    model.add(TimeDistributed(Dense(N_FEATURES)))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    return model
{% endhighlight %}


### Training

Training is easy. I'll run batches of 10 epochs and then print out some
example predictions. Here's the main function of our code:

{% highlight python %}
from time import sleep

def main():
    model = build_model()

    model.summary()
    print()

    x_test, y_test, x_train, y_train = build_dataset()

    # Let's print some predictions now to get a feeling for the equations
    print()
    print_example_predictions(5, model, x_test, y_test)
    print()

    try:
        for iteration in range(int(EPOCHS / 10)):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            model.fit(
                x_train, y_train,
                epochs=10,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
            )
            sleep(0.01)

            print()
            print_example_predictions(5, model, x_test, y_test)
            print()

    except KeyboardInterrupt:
        print(' Got Sigint')
    finally:
        sleep(0.01)
        model.save('model.h5')

        print_example_predictions(20, model, x_test, y_test)


if __name__ == '__main__':
    main()
{% endhighlight %}

Finally, we need to fill in some of the global variables we used throughout
the code. With two numbers from 0 to 999, there's 998,001 possible equations.
Let's use about a sixteenth of that, meaning we'll train on 31,188 and validate
on 31,187. Here's my config:

{% highlight python %}
MIN_NUMBER = 0
MAX_NUMBER = 999

MAX_N_EXAMPLES = (MAX_NUMBER - MIN_NUMBER) ** 2
N_EXAMPLES = int(round(MAX_N_EXAMPLES / 16.))
N_FEATURES = 12
MAX_NUMBER_LENGTH_LEFT_SIDE = max(len(str(MAX_NUMBER)), len(str(MIN_NUMBER)))
MAX_NUMBER_LENGTH_RIGHT_SIDE = MAX_NUMBER_LENGTH_LEFT_SIDE + 1
MAX_EQUATION_LENGTH = (MAX_NUMBER_LENGTH_LEFT_SIDE * 2) + 3
MAX_RESULT_LENGTH = MAX_NUMBER_LENGTH_RIGHT_SIDE

SPLIT = .5
EPOCHS = 800
BATCH_SIZE = 128
{% endhighlight %}

Finally, we can start training. Either call `main()` from the shell, or store
everything in a file `training.py` and run it via `python training.py`.


### Results

Running the code as it is described here, I get to an accuracy of `.95` on the
test set after 120 epochs.

Let's see how the training evolved:

##### Before training

```
Example predictions:
808 + 570 = 9990   (expected: 1378)
25 + 941  = 3333   (expected:  966)
230 + 331 = 2222   (expected:  561)
502 + 799 = 1111   (expected: 1301)
17 + 108  = 7777   (expected:  125)
```

As expected, the output is pretty random.

##### Epoch 10

```
Validation loss:      1.339
Validation accuracy:  0.486

Example predictions:
454 + 873 = 1399   (expected: 1327)
573 + 106 =  679   (expected:  679)
501 + 423 =  999   (expected:  924)
782 + 494 = 1299   (expected: 1276)
900 + 512 = 1479   (expected: 1412)
```

Wow, already got the basic idea of numbers and especially 3-digit vs 4-digit.
One example is spot-on even.

##### Epoch 50

```
Validation loss:      0.931
Validation accuracy:  0.637

Example predictions:
220 + 642 =  865   (expected:  862)
710 + 426 = 1139   (expected: 1136)
231 + 901 = 1144   (expected: 1132)
822 + 666 = 1488   (expected: 1488)
915 + 565 = 1470   (expected: 1480)
```

We're getting there! The numbers are generally right except for some single
digits.

##### Epoch 150

```
Validation loss:      0.116
Validation accuracy:  0.961

Example predictions:
726 + 639 = 1365   (expected: 1365)
807 + 373 = 1180   (expected: 1180)
68 + 852  =  920   (expected:  920)
364 + 495 =  859   (expected:  859)
914 + 743 = 1657   (expected: 1657)
```

It started overfitting here, so I stopped. As you see, the results are
basically perfect, though the accuracy of 96% means we're still doing some
small mistakes from time to time.




### Analysis

TODO Add analysis of single neurons, etc.


### Further Experiments

There's lots to do! Of course we can increase the complexity of the equations.
I was able to get quite for on ones as complex as
`131 + 83 - 744 * 33 (= -24338)`, but haven't really gotten it to work with
division, or anything with decimals at all. Usually that either means the
model is not complex enough (try increasing the hidden units or depth of
encoder and decoder) or we simply could use some more training examples. Maybe
also the training process can be improved, e.g. by regularization techniques
like dropout or batch normalization. It's quite silly that we get up to 96%
accuracy and then plateau... I'm sure with more tweaking and patience, this
can be improved.

Feel free to pass on hints, ideas for improvement, or your own results in the
comments or as issues on my [repository](https://github.com/cpury/lstm-math).
