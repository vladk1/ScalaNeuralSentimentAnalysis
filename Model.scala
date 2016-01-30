package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.numerics.sigmoid
import uk.ac.ucl.cs.mr.statnlpbook.assignment3._
import breeze.linalg._
import ml.wolfe.nlp.{SentenceSplitter, TokenSplitter}

import scala.collection.mutable

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
    calculateLossNormally(sentence, target)
  }

  def calculateLossNormally(sentence: Seq[String], target: Boolean): Loss = {
    val targetScore = if (target) 1.0 else 0.0
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    val score = scoreSentence(sentenceVector)
    new LossSum(NegativeLogLikelihoodLoss(score, targetScore), regularizer(wordVectors))
  }


  def isGood(s: String): Boolean = {
    if ((s.size <= 2) ||
      (s.charAt(0) == '@') ||
      (s.matches("[\\p{Punct}\\s]+") && !s.matches("!")) ||
      s.matches("[0-9]") ) false
    else true
    //       s.matches("[^\\w_\\s]+") || //to check non-english words
  }
  def preprocessInput(s: Seq[String]): Seq[String] = {
    val noLinks = s.filterNot(word =>  word.contains("http") || word.contains("www")).mkString(" ")
    val filteredTokenizedSentence = SentenceSplitter(TokenSplitter(noLinks)).tokenWords.filter(word => isGood(word)).slice(0, 6)
    filteredTokenizedSentence
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
  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors
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
    words.foldLeft(vectorParams("param_h0"):Block[Vector])((h_prev, wordVector) => {
      Tanh(Sum(Seq(
        Mul(matrixParams("param_Wh"), h_prev), Mul(matrixParams("param_Wx"), wordVector), vectorParams("param_b")
      )))
    })
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
     Sigmoid(Dot(sentence, vectorParams("param_w")))
  }

  def regularizer(words: Seq[Block[Vector]]): Loss = {
    val l2matrixregul = Seq(matrixParams("param_Wh"), matrixParams("param_Wx"))
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_h0") :+ vectorParams("param_b"): _*),
      L2Regularization(matrixRegularizationStrength, l2matrixregul: _*)
    )
  }
}

/**
  * Problem 4
  * A LSTM model
  * Input gate parameters:
  *
  *
  * @param embeddingSize dimension of the word vectors used in this model
  * @param hiddenSize dimension of the hidden state vector used in this model
  * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
  * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
  */
class LSTMModel(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {
  println(s"LSTMModel: embedSize=$embeddingSize hiddenSize=$hiddenSize vectorReg=$vectorRegularizationStrength " +
    s"matrixReg=$matrixRegularizationStrength \n")
  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors
  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()

  val vectorParamLabels = Seq("param_w", "param_h0", "param_c0", "param_b_i", "param_b_f", "param_b_o", "param_b_g")
  vectorParamLabels.foreach(paramLabel => {
    vectorParams += paramLabel -> VectorParam(hiddenSize)
  })

  val matrixEmbeddLabels = Seq("param_W_i", "param_W_f", "param_W_o", "param_W_g")
  matrixEmbeddLabels.foreach(paramLabel => {
    matrixParams += paramLabel -> MatrixParam(hiddenSize, embeddingSize)
  })

  val matrixHiddenLabels = Seq("param_H_i", "param_H_f", "param_H_o", "param_H_g")
  matrixHiddenLabels.foreach(paramLabel => {
    matrixParams += paramLabel -> MatrixParam(hiddenSize, hiddenSize)
  })

  val biasLabels = Seq("param_b_i", "param_b_f", "param_b_g", "param_b_o")
  // Initialize bias params to zero
  biasLabels.foreach(paramLabel => {
    vectorParams(paramLabel).set(DenseVector.zeros[Double](hiddenSize))
  })

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  override def predict(sentence: Seq[String])(implicit threshold: Double = 0.5):Boolean = {
    val filteredTokenizedSentence = preprocessInput(sentence)
    val wordVectors = filteredTokenizedSentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    scoreSentence(sentenceVector).forward() >= threshold
  }

  override def loss(sentence: Seq[String], target: Boolean): Loss = {
    val filteredSentence = preprocessInput(sentence)
    calculateLossNormally(filteredSentence, target)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
    val h0 = vectorParams("param_h0"):Block[Vector]
    val c0 = vectorParams("param_c0"):Block[Vector]
    val W_i = matrixParams("param_W_i")
    val W_g = matrixParams("param_W_g")
    val W_f = matrixParams("param_W_f")
    val W_o = matrixParams("param_W_o")

    val H_i = matrixParams("param_H_i")
    val H_g = matrixParams("param_H_g")
    val H_f = matrixParams("param_H_f")
    val H_o = matrixParams("param_H_o")

    val b_i = vectorParams("param_b_i")
    val b_g = vectorParams("param_b_g")
    val b_f = vectorParams("param_b_f")
    val b_o = vectorParams("param_b_o")

    val h_n = words.foldLeft((h0,c0))((prev, x_t) => {

      val i = VectorSigmoid(Sum(Seq(Mul(W_i, x_t), Mul(H_i, prev._1), b_i)))  // i = sigmoid( Wi⋅xt + Hi⋅h_(t-1) + bi )

      val g = Tanh(Sum(Seq(Mul(W_g, x_t), Mul(H_g, prev._1), b_g)))           // g = tanh( Wg⋅xt + Hg⋅h_(t-1) + bg )

      val f = VectorSigmoid(Sum(Seq(Mul(W_f, x_t), Mul(H_f, prev._1), b_f)))  // f = sigmoid( Wf⋅xt + Hf⋅h_(t-1) + bf )

      val c_t = Sum(Seq(ElementMul(Seq(i, g)), ElementMul(Seq(f, prev._2))))  // c = i * g + f * c_(t-1)

      val o = VectorSigmoid(Sum(Seq(Mul(W_o, x_t), Mul(H_o, prev._1), b_o)))  // o = sigmoid( Wo⋅xt + Ho⋅h_(t-1) + bo )

      val h_t = ElementMul(Seq(Tanh(c_t), o))                                 // h = tanh(c) * o

      (h_t, c_t)
    })
    h_n._1
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    val output = Sigmoid(Dot(sentence, vectorParams("param_w")))
    output
  }

  def regularizer(words: Seq[Block[Vector]]): Loss = {

    val l2VectorArgs = (vectorParamLabels ++ biasLabels).foldLeft(words)((args, paramLabel) => {
      args :+ vectorParams(paramLabel)
    })
//    use to remove word embeddings
//    val l2VectorArgs = (vectorParamLabels ++ biasLabels).map(paramLabel => {
//      vectorParams(paramLabel)
//    })
    val l2MatrixArgs = (matrixEmbeddLabels ++ matrixHiddenLabels).map(paramLabel => {
      matrixParams(paramLabel)
    })

    new LossSum(
      L2Regularization(vectorRegularizationStrength, l2VectorArgs: _*),
      L2Regularization(matrixRegularizationStrength, l2MatrixArgs: _*)
    )
  }
}


class GRUModel(embeddingSize: Int, hiddenSize: Int,
                vectorRegularizationStrength: Double = 0.0,
                matrixRegularizationStrength: Double = 0.0) extends Model {
  println(s"GRUModel: embedSize=$embeddingSize hiddenSize=$hiddenSize vectorReg=$vectorRegularizationStrength " +
    s"matrixReg=$matrixRegularizationStrength \n")

  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors
  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()

  val vectorParamLabels = Seq("param_w", "param_h0", "param_b_z", "param_b_r", "param_b_g")
  vectorParamLabels.foreach(paramLabel => {
    vectorParams += paramLabel -> VectorParam(hiddenSize)
  })

  val matrixEmbeddLabels = Seq("param_W_z", "param_W_r", "param_W_g")
  matrixEmbeddLabels.foreach(paramLabel => {
    matrixParams += paramLabel -> MatrixParam(hiddenSize, embeddingSize)
  })

  val matrixHiddenLabels = Seq("param_H_z", "param_H_r", "param_H_g")
  matrixHiddenLabels.foreach(paramLabel => {
    matrixParams += paramLabel -> MatrixParam(hiddenSize, hiddenSize)
  })

  val biasLabels = Seq("param_b_z", "param_b_r", "param_b_g")
  // Initialize bias params to zero
  biasLabels.foreach(paramLabel => {
    vectorParams(paramLabel).set(DenseVector.zeros[Double](hiddenSize))
  })

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  override def predict(sentence: Seq[String])(implicit threshold: Double = 0.5):Boolean = {
    val filteredTokenizedSentence = preprocessInput(sentence)
    val wordVectors = filteredTokenizedSentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    scoreSentence(sentenceVector).forward() >= threshold
  }

  override def loss(sentence: Seq[String], target: Boolean): Loss = {
    val filteredSentence = preprocessInput(sentence)
    calculateLossNormally(filteredSentence, target)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
    val h0 = vectorParams("param_h0"):Block[Vector]
    val b_z = vectorParams("param_b_z")
    val b_r = vectorParams("param_b_r")
    val b_g = vectorParams("param_b_g")

    val W_z = matrixParams("param_W_z")
    val W_r = matrixParams("param_W_r")
    val W_g = matrixParams("param_W_g")
    val H_z = matrixParams("param_H_z")
    val H_r = matrixParams("param_H_r")
    val H_g = matrixParams("param_H_g")

    val h_n = words.foldLeft(h0)((prev_h, x_t) => {

      val z = VectorSigmoid(Sum(Seq(Mul(W_z, x_t), Mul(H_z, prev_h), b_z)))
      val r = VectorSigmoid(Sum(Seq(Mul(W_r, x_t), Mul(H_r, prev_h), b_r)))

      val gg = ElementMul(Seq(r, prev_h))
      val g = Tanh(Sum(Seq(Mul(W_g, x_t), Mul(H_g, gg), b_g)))

      val ones = vec((0 until hiddenSize).map(i => 1.0):_*)
      val zsub = Sub(Seq(ones, z))
      val h_t = Sum(Seq(ElementMul(Seq(zsub, prev_h)), ElementMul(Seq(z, g))))

      h_t
    })

    h_n
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    val output = Sigmoid(Dot(sentence, vectorParams("param_w")))
    output
  }

  def regularizer(words: Seq[Block[Vector]]): Loss = {

    val l2VectorArgs = (vectorParamLabels ++ biasLabels).foldLeft(words)((args, paramLabel) => {
      args :+ vectorParams(paramLabel)
    })

//    use to remove word embeddings
//    val l2VectorArgs = (vectorParamLabels ++ biasLabels).map(paramLabel => {
//      vectorParams(paramLabel)
//    })

    val l2MatrixArgs = (matrixEmbeddLabels ++ matrixHiddenLabels).map( paramLabel => {
      matrixParams(paramLabel)
    })

    new LossSum(
      L2Regularization(vectorRegularizationStrength, l2VectorArgs: _*),
      L2Regularization(matrixRegularizationStrength, l2MatrixArgs: _*)
    )
  }
}



/**
  * Problem 4 - Second approach
  * A combination between multiplication and sum of word vectors model
  *
  * @param embeddingSize dimension of the word vectors used in this model
  * @param regularizationStrength strength of the regularization on the word vectors and global parameter vector w
  */
class MulOfWordsModel(embeddingSize: Int, regularizationStrength: Double = 0.001) extends Model {
  /**
    * We use a lookup table to keep track of the word representations
    */
  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors

  vectorParams += "param_w" -> VectorParam(embeddingSize)
  vectorParams += "param_bias" -> VectorParam(embeddingSize)

  var init = DenseVector.zeros[Double](embeddingSize)
  init(0 to embeddingSize-1) := 1.0 / embeddingSize
  vectorParams("param_bias").set(init)

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
    Sum(Seq(ElementMul(words), Sum(words), vectorParams("param_bias")))
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(sentence, vectorParams("param_w")))

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization(regularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_bias") :_*)

}

