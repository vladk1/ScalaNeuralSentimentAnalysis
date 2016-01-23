package uk.ac.ucl.cs.mr.statnlpbook.assignment3

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


  def regularizer(words: Seq[Block[Vector]]): Loss = {
    L2Regularization(regularizationStrength, wordVectorsToSentenceVector(words))
  }
//  words :+ vectorParams("param_w") :_*
//    L2Regularization(regularizationStrength, wordVectorsToSentenceVector(words))
}
//words :+ vectorPar :_*

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
  vectorParams += "param_w" -> VectorParam(hiddenSize)
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  vectorParams += "param_b" -> VectorParam(hiddenSize)

  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_Wx" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_Wh" -> MatrixParam(hiddenSize, hiddenSize)

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
    // convert words to indexed seq
    val indexedWords = words.toIndexedSeq
    // create placeholder vector for sentence

    var sentence = DenseVector.zeros[Double](words.size * embeddingSize)
    sentence.forward()
//    println("print senetence init" + sentence.ou)
    // loop through words and use logic index to move words to index in sentence vector
    for (i <- 0 until words.size) {
//      println("word size" + words.size)
      val start = i * embeddingSize
      val end = start + embeddingSize
      val word = indexedWords(i).forward()
//      println(start + " " + end)
//      println(sentence(start until end))
      sentence(start until end) := indexedWords(i).forward()
//      println()
//      println("eq" + word == sentence(start until end))
//      println(word)
//      println(sentence(start until end))
    }

//    println("sentence " + sentence)
    sentence
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    // make initial copy of h0
//    var ht = vectorParams("param_h0")
//    vectorParams("param_b").set(vec(DenseVector.zeros[Double](hiddenSize).toArray.toSeq:_*))
//    vectorParams("param_b").forward()
//    val h0 = DenseVector.zeros[Double](hiddenSize)
    var ht = VectorParam(hiddenSize)
//    ht.set(vec(DenseVector.zeros[Double](hiddenSize).toArray.toSeq:_*))
//    ht.forward()
//    println(ht.output)
//    println("init ht" + ht.forward())
    // loop through each words
    sentence.forward()
    val numWords = (sentence.output.activeSize / embeddingSize)
//    print("words " + numWords)
    for(i <- 0 until numWords) {
//      println("num words" + sentence.forward().activeSize / embeddingSize)
      // set start and end index to retrieve word vector
      val start = i * embeddingSize
      val end = start + embeddingSize
      val xt = sentence.output(start until end)
//      println(ht.output)
      // perform ht calculation
      val Whhtprev = Mul(matrixParams("param_Wh"), ht)
      val Wxxt = Mul(matrixParams("param_Wx"), xt)
      val b = vectorParams("param_b")
      val htupdate = Tanh(Sum(Seq(Whhtprev, Wxxt, b)))
      // update ht param
//      print(ht.output)
//      println("old ht" + ht.forward())
//      println("new score" + score.forward())
      ht.set(htupdate.forward())
      ht.forward()

//      println("new ht" + ht.forward())
    }
    // set h_n as sentence representation
//    println("final ht" + ht.forward())
//    println("size of ht = " + ht.gradParam.activeSize + "   " + ht.param.activeSize)
//    println("dot ht ht" + Dot(ht,ht).forward())
//    println("dot ht b" + Dot(ht,vectorParams("param_b")).forward())
//    println
    val sentenceScore = Sigmoid(Dot(ht, vectorParams("param_w")))
//    println(sentenceScore.forward())
//    sentenceScore
//    sentenceScore.forward()

//    val sentenceScore = Sigmoid(DoubleConstant(sum(ht.output) / hiddenSize))
    println(sentenceScore.forward())
    sentenceScore
//    Sigmoid(Dot(ht, vectorParams()))
  }

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, vectorParams("param_w")),
      L2Regularization(matrixRegularizationStrength, matrixParams("param_Wx"))
//      L2Regularization(vectorRegularizationStrength, wordVectorsToSentenceVector(words)),
//      L2Regularization(matrixRegularizationStrength, wordVectorsToSentenceVector(words))
    )
}