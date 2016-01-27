package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.numerics.sigmoid

import scala.collection.mutable
import breeze.linalg._

/**
 * @author rockt
 */
trait Model {
  /**
   * Stores all vector parameters
   */
  val vectorParams = new mutable.HashMap[String, VectorParam]()
  /**
   * Stores all matrix parameters
   */
  val matrixParams = new mutable.HashMap[String, MatrixParam]()
  /**
   * Maps a word to its trainable or fixed vector representation
    *
    * @param word the input word represented as string
   * @return a block that evaluates to a vector/embedding for that word
   */
  def wordToVector(word: String): Block[Vector]
  /**
   * Composes a sequence of word vectors to a sentence vectors
    *
    * @param words a sequence of blocks that evaluate to word vectors
   * @return a block evaluating to a sentence vector
   */
  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector]
  /**
   * Calculates the score of a sentence based on the vector representation of that sentence
    *
    * @param sentence a block evaluating to a sentence vector
   * @return a block evaluating to the score between 0.0 and 1.0 of that sentence (1.0 positive sentiment, 0.0 negative sentiment)
   */
  def scoreSentence(sentence: Block[Vector]): Block[Double]
  /**
   * Predicts whether a sentence is of positive or negative sentiment (true: positive, false: negative)
    *
    * @param sentence a tweet as a sequence of words
   * @param threshold the value above which we predict positive sentiment
   * @return whether the sentence is of positive sentiment
   */
  def predict(sentence: Seq[String])(implicit threshold: Double = 0.5): Boolean = {
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    scoreSentence(sentenceVector).forward() >= threshold
  }
  /**
   * Defines the training loss
    *
    * @param sentence a tweet as a sequence of words
   * @param target the gold label of the tweet (true: positive sentiement, false: negative sentiment)
   * @return a block evaluating to the negative log-likelihood plus a regularization term
   */
  def loss(sentence: Seq[String], target: Boolean): Loss = {
    val targetScore = if (target) 1.0 else 0.0
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    val score = scoreSentence(sentenceVector)
    new LossSum(NegativeLogLikelihoodLoss(score, targetScore), regularizer(wordVectors))
  }
  /**
   * Regularizes the parameters of the model for a given input example
    *
    * @param words a sequence of blocks evaluating to word vectors
   * @return a block representing the regularization loss on the parameters of the model
   */
  def regularizer(words: Seq[Block[Vector]]): Loss
}


/**
 * Problem 2
 * A sum of word vectors model
  *
  * @param embeddingSize dimension of the word vectors used in this model
 * @param regularizationStrength strength of the regularization on the word vectors and global parameter vector w
 */
class SumOfWordVectorsModel(embeddingSize: Int, regularizationStrength: Double = 1000.0) extends Model {
  /**
   * We use a lookup table to keep track of the word representations
   */
  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors

  /**
   * We are also going to need another global vector parameter
   */
  vectorParams += "param_w" -> VectorParam(embeddingSize)

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Sum(words)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(sentence, vectorParams("param_w")))

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization(regularizationStrength, words :+ vectorParams("param_w") :_*)

}

/**
 * Problem 3
 * A recurrent neural network model
  *
  * @param embeddingSize dimension of the word vectors used in this model
 * @param hiddenSize dimension of the hidden state vector used in this model
 * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
 * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
 */
class RecurrentNeuralNetworkModel(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  vectorParams += "param_w" -> VectorParam(hiddenSize) // supposed to be embeddingSize, currently hiddenSize
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  vectorParams += "param_b" -> VectorParam(hiddenSize)

  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_Wx" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_Wh" -> MatrixParam(hiddenSize, hiddenSize)

  // Initialize parameters
  vectorParams("param_b").set(DenseVector.zeros[Double](hiddenSize))

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
    words.foldLeft(vectorParams("param_w"):Block[Vector])((h_prev, wordVector) => {
      val Wh_h_prev = Mul(matrixParams("param_Wh"), h_prev)
      val Wx_x_t = Mul(matrixParams("param_Wx"), wordVector)
      val b = vectorParams("param_b")
      val new_h = Tanh(Sum(Seq(Wh_h_prev, Wx_x_t, b)))
      new_h
    })
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
      Sigmoid(Dot(sentence, vectorParams("param_w")))
  }

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
        L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_h0") :+ vectorParams("param_b"):_*),
        L2Regularization(matrixRegularizationStrength, words :+ matrixParams("param_Wh") :+ matrixParams("param_Wx"):_*)
    )
}
//vectorParams += "param_w" -> VectorParam(hiddenSize) // supposed to be embeddingSize, currently hiddenSize
//vectorParams += "param_h0" -> VectorParam(hiddenSize)
//vectorParams += "param_b" -> VectorParam(hiddenSize)
/**
  * Problem 4
  * A LSTM model
  * Input gate parameters:
  *
  * @param
  * @param embeddingSize dimension of the word vectors used in this model
  * @param hiddenSize dimension of the hidden state vector used in this model
  * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
  * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
  */
class LSTMModel(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
//  vectorParams += "param_w" -> VectorParam(hiddenSize) // supposed to be embeddingSize, currently hiddenSize
//  vectorParams += "param_h0" -> VectorParam(hiddenSize)
//  vectorParams += "param_b" -> VectorParam(hiddenSize)
  vectorParams += "param_w" -> VectorParam(hiddenSize)
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  vectorParams += "param_c0" -> VectorParam(hiddenSize)
  vectorParams += "param_b_i" -> VectorParam(hiddenSize)
  vectorParams += "param_b_f" -> VectorParam(hiddenSize)
  vectorParams += "param_b_g" -> VectorParam(hiddenSize)
  vectorParams += "param_b_o" -> VectorParam(hiddenSize)

  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_W_i" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_H_i" -> MatrixParam(hiddenSize, hiddenSize)
  matrixParams += "param_W_f" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_H_f" -> MatrixParam(hiddenSize, hiddenSize)
  matrixParams += "param_W_o" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_H_o" -> MatrixParam(hiddenSize, hiddenSize)
  matrixParams += "param_W_g" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_H_g" -> MatrixParam(hiddenSize, hiddenSize)

  // Initialize parameters
  vectorParams("param_b_i").set(DenseVector.zeros[Double](hiddenSize))
  vectorParams("param_b_f").set(DenseVector.zeros[Double](hiddenSize))
  vectorParams("param_b_g").set(DenseVector.zeros[Double](hiddenSize))
  vectorParams("param_b_o").set(DenseVector.zeros[Double](hiddenSize))

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
//    println(words.size)
    val h_n = words.foldLeft((vectorParams("param_h0"):Block[Vector],vectorParams("param_c0"):Block[Vector]))((hc, x_t) => {
      val i = VectorSigmoid(Sum(Seq(Mul(matrixParams("param_W_i"), x_t), Mul(matrixParams("param_H_i"), hc._1), vectorParams("param_b_i"))))
      val f = VectorSigmoid(Sum(Seq(Mul(matrixParams("param_W_f"), x_t), Mul(matrixParams("param_H_f"), hc._1), vectorParams("param_b_f"))))
      val o = VectorSigmoid(Sum(Seq(Mul(matrixParams("param_W_o"), x_t), Mul(matrixParams("param_H_o"), hc._1), vectorParams("param_b_o"))))
      val g = Tanh(Sum(Seq(Mul(matrixParams("param_W_g"), x_t), Mul(matrixParams("param_H_g"), hc._1), vectorParams("param_b_g"))))
      val c_t = Sum(Seq(ElementMul(Seq(hc._2, f)), ElementMul(Seq(g, i))))
      val h_t = ElementMul(Seq(Tanh(c_t), o))
//      println(h_t)
      (h_t, c_t)
    })
//    println(h_n._1)
    h_n._1
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    print(".")
    val output = Sigmoid(Dot(sentence, vectorParams("param_w")))
    println('.')
    output
  }

//  def regularizer(words: Seq[Block[Vector]]): Loss =
//    new LossSum(
//      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_h0"):_*),
//      L2Regularization(matrixRegularizationStrength, words :+ matrixParams("param_W_i") :+ matrixParams("param_H_i")
//                                                           :+ matrixParams("param_W_f") :+ matrixParams("param_H_f")
//                                                           :+ matrixParams("param_W_o") :+ matrixParams("param_H_o")
//                                                           :+ matrixParams("param_W_g") :+ matrixParams("param_H_g"):_*)
//    )

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_h0")
              :+vectorParams("param_b_i"):+vectorParams("param_b_f"):+vectorParams("param_b_g"):+vectorParams("param_b_o"):_*),
      L2Regularization(matrixRegularizationStrength, words :+ matrixParams("param_W_i") :+ matrixParams("param_H_i")
                                                    :+ matrixParams("param_W_f") :+ matrixParams("param_H_f")
                                                    :+ matrixParams("param_W_o") :+ matrixParams("param_H_o")
                                                    :+ matrixParams("param_W_g") :+ matrixParams("param_H_g"):_*)
    )
}
