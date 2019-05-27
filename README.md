# AI_CNN_project
Porject for the Artificial Intelligence Course FRI Ljubljana 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2345678901234567890123456789012345678901234567890123456789012345678901234567890
%        1         2         3         4         5         6         7         8

\documentclass[letterpaper, 12 pt, conference]{ieeeconf}  % Comment this line out
                                                          % if you need a4paper
%\documentclass[a4paper, 12pt, conference]{ieeeconf}      % Use this line for a4
                                                          % paper

\IEEEoverridecommandlockouts                              % This command is only
                                                          % needed if you want to
                                                          % use the \thanks command
\overrideIEEEmargins
% See the \addtolength command later in the file to balance the column lengths
% on the last page of the document
\usepackage{graphicx}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

% The following packages can be found on http:\\www.ctan.org
%\usepackage{graphics} % for pdf, bitmapped graphics files
%\usepackage{epsfig} % for postscript graphics files
%\usepackage{mathptmx} % assumes new font selection scheme installed
%\usepackage{times} % assumes new font selection scheme installed
%\usepackage{amsmath} % assumes amsmath package installed
%\usepackage{amssymb}  % assumes amsmath package installed

\title{\LARGE \bf
Image Processing using Convolutional Neural Network (CNN)
}

%\author{ \parbox{3 in}{\centering Narshion Ngao*
%         \thanks{*Use the $\backslash$thanks command to put information here}\\
%         Msc. Computer Systems - 2018\\
%         Jomo Kenyatta University of Agriculture \& Technology \\
%       
%}}

\author{Brieuc Vably, Ilaria Lauriola \\% <-this % stops a space
University of Ljubljana \\
Faculty of Computer and Information Science \\
Course: Artificial Intelligence
}


\begin{document}



\maketitle
\thispagestyle{empty}
\pagestyle{empty}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{abstract}

This is a report for the project of Artificial Intelligence for image analysis: recognition and classification using using CNN, Keras and Tensorflow backend. In this report we are explaning a little bit of theory behind CNN, the steps behind the algorythm and some results and conclusions.

\end{abstract}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Convolutional Neural Networks (CNN)}
\subsection{\textbf{Overview}}

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
\\A ConvNet is able to successfully capture the Spatial and Temporal dependencies in an image through the application of relevant filters. The architecture performs a better fitting to the image dataset due to the reduction in the number of parameters involved and reusability of weights.
\subsection{\textbf{How to build a CNN}}
There are four main operations used to build blocks in the CNN, summed up as follow:

\subsubsection{Convolution}
Convolution purpose is to extract features from the input image. Convolution is a linear operation and preserves the spatial relationship between pixels by learning image features using small squares of input data.
\subsubsection{Non Linearity (ReLU)}
ReLU stands for Rectified Linear Unit and is a non-linear operation.
ReLU is an element wise operation (applied per pixel) and replaces all negative pixel values in the feature map by zero. The purpose of ReLU is to introduce non-linearity in our CNN.
\subsubsection{Pooling or Sub Sampling}
Spatial Pooling (also called subsampling or downsampling) reduces the dimensionality of each feature map but retains the most important information. Spatial Pooling can be of different types: Max, Average, Sum etc.
\subsubsection{Classification (Fully Connected Layer)}
The Fully Connected layer is a traditional Multi Layer Perceptron that uses a softmax activation function in the output layer. His purpose is to use these features for classifying the input image into various classes based on the training dataset (trained using Backpropagation).

\begin{figure}
	\includegraphics[width=\columnwidth]{cnn.png}
    \caption{CNN}
    \label{fig:example_figure}
\end{figure}

\subsection{\textbf{Implementation}}
In this section we briefly explain how we build the algorithm for image analysis based of the steps described before.

\subsubsection{\textbf{Case Study}}
The main problem in this case is to find a huge dataset that can be used to train the model.
We chose the one that can explain in a simple but efficient way how CNN works, the dog and cats dataset, containing 10000 images, 5000 of each.
Tensorflow and Keras are powerful open source libraries used to build our classifier.
Then all the steps described before have to be implemented while writing the code, using Python as programming language and Spider as IDE.
First thing to be done, is to split the dataset in training and test set. Usually the ratio is 80\% train and 20\% set, and that is what we did in our case.
Sigmoid function has been used as an activation function for the last NN layers as we need to find the probability of the object being a cat or a dog.
While building the CNN we set the number of epoch as 25 and 8000 steps per epoch and trained our model. Incresing the number of epoch also the accuracy will increase. We will discuss it later while evaluating the classifier.
Choosing these values, training with a 32giga ram device with Nvidia GeForce GTX 1050 video card it took 2h15m to finish the training task.

\subsubsection{\textbf{Testing the Classifier}}
The testing part was made using a random image from test set, first choosing dogs and then cats.



\subsubsection{\textbf{Evaluating the Classifier}}
 Adding more convolutional and pooling layers, playing with the number of nodes and epochs, might help getting high accuracy result.
 Some evaluating methods that can be used to evaluate the classifier are:
 \begin{itemize}
     \item Accuracy classification score.
     \item Confusion Matrix.
     \item Cross-entropy loss.
     \item K-Fold Cross Validation
 \end{itemize}

We are going to use confusion matrix.
We tested the algorithm on 8 images, 4 of dogs and 4 of cats.
The dogs identified as 1 value, the cats as 0.
All the dogs images where correctly predicted while only 2 images of cats where correctly classified.
We build the confusion matrix using the sklearn function.
What we got is this:
\begin{table}
	\centering
	\caption{Confusion Matrix.}
	\label{tab:example_table}
	\begin{tabular}{lccr} % four columns, alignment for each
		\hline
		 & P & N \\
		\hline
		P & 2 & 2 \\
		N & 0 & 4 \\
	
		\hline
	\end{tabular}
\end{table}
The first row value 2 is the values of cat's images correctly classified, the second value is cat's images wrong classified, going to the second row the 0 means dog's images wrong classified and 4 is dog's images right classified.
We also evaluated the precision, recall, f1 score and support.
\begin{figure}
	\includegraphics[width=\columnwidth]{img.PNG}
    \caption{Evaluation scores}
    \label{fig:example_figure}
\end{figure}

\begin{figure}
	\includegraphics[scale=0.020,width=\columnwidth]{matrix.png}
    \caption{Confusion Matrix}
    \label{fig:example_figure}
\end{figure}


Although it is only a small test on only few test datas, done just to see how our classifier works, the values we got are quite good, but still can be significantly improved.
\section{CONCLUSIONS}

The field of artificial intelligence is gaining momentum especially in this new era of advanced computing. Even if this CNN is used as simple classifier various fields are now taking advantage of this image processing method, expecially in medical cancer research.

\addtolength{\textheight}{-12cm}   % This command serves to balance the column lengths
                                  % on the last page of the document manually. It shortens
                                  % the textheight of the last page by a suitable amount.
                                  % This command does not take effect until the next page
                                  % so it should come on the page before the last. Make
                                  % sure that you do not shorten the textheight too much.



\begin{thebibliography}{99}

\bibitem{c1} https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

\end{thebibliography}




\end{document}
