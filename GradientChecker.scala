package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.{QuasiTensor, TensorLike, sum}
import breeze.numerics._

/**
 * Problem 1
 */
object GradientChecker extends App {
  val EPSILON = 1e-6

  /**
   * For an introduction see http://cs231n.github.io/neural-networks-3/#gradcheck
   *
   * This is a basic implementation of gradient checking.
   * It is restricted in that it assumes that the function to test evaluates to a double.
   * Moreover, another constraint is that it always tests by backpropagating a gradient of 1.0.
   */
  def apply[P](model: Block[Double], paramBlock: ParamBlock[P]) = {
    paramBlock.resetGradient()
    model.forward()
    model.backward(1.0)

    var avgError = 0.0

    val gradient = paramBlock.gradParam match {
      case m: Matrix => m.toDenseVector
      case v: Vector => v
    }

    /**
     * Calculates f_theta(x_i + eps)
     * @param index i in x_i
     * @param eps value that is added to x_i
     * @return
     */
    def wiggledForward(index: Int, eps: Double): Double = {
      var result = 0.0
      paramBlock.param match {
        case v: Vector =>
          val tmp = v(index)
          v(index) = tmp + eps
          result = model.forward()
          v(index) = tmp
        case m: Matrix =>
          val (row, col) = m.rowColumnFromLinearIndex(index)
          val tmp = m(row, col)
          m(row, col) = tmp + eps
          result = model.forward()
          m(row, col) = tmp
      }
      result
    }

    for (i <- 0 until gradient.activeSize) {
      //todo: your code goes here!
      val gradientExpected: Double = ( wiggledForward(i, EPSILON) - wiggledForward(i, -EPSILON) ) / ( 2 * EPSILON )

      avgError = avgError + math.abs(gradientExpected - gradient(i))

      assert(
        math.abs(gradientExpected - gradient(i)) < EPSILON,
        "Gradient check failed!\n" +
          s"Expected gradient for ${i}th component in input is $gradientExpected but I got ${gradient(i)}"
      )
    }
    println("Average error: " + avgError)
  }

  /**
    * A very silly block to test if gradient checking is working.
    * Will only work if the implementation of the Dot block is already correct
    */

    // GRADIENT CHECKING - INITIALIZATION
    val a = vec(-1.5, 1.0, 1.5, 0.5)
    val c = vec(1.0, 2.0, 5.0, 2.5)
    val b = VectorParam(4); b.set(vec(1.0, 2.0, -0.5, 2.5))
    val matrix = MatrixParam(4,4)

    // DOT BLOCK
    val simpleDotBlock = Dot(a, b)
    println("\nGradient checking DOT block wrt vectorParam: ")
    GradientChecker(simpleDotBlock, b)

    //  SUM BLOCK
    val sumBlock = Sum(Seq(a, b))
    println("\nGradient checking SUM block wrt vectorParam: ")
    GradientChecker(Dot(sumBlock, b), b)

    // SIGMOID BLOCK
    val sigmoidBlock = Sigmoid(Dot(a, b))
    println("\nGradient checking SIGMOID block wrt vectorParam: ")
    GradientChecker(sigmoidBlock, b)

    //  NLL BLOCK
    val negativeLogLikelihoodLossBlock = NegativeLogLikelihoodLoss(Sigmoid(Dot(a, b)), 0.5)
    println("\nGradient checking NEGATIVE LOG LIKELIHOOD block wrt vectorParam: ")
    GradientChecker(negativeLogLikelihoodLossBlock, b)

    // MUL BLOCK
    val mulBlock = Mul(matrix, c)
    println("\nGradient checking MUL block wrt vectorParam: ")
    GradientChecker(Dot(mulBlock, b), b)
//    GradientChecker(Dot(mulBlock,mulBlock), matrix)

    // TANH BLOCK
    val tanBlock = Tanh(b)
    println("\nGradient Checking TANH block wrt vectorParam: ")
    GradientChecker(Dot(tanBlock, tanBlock), b)

    // L2 REGULARIZATION BLOCK
    val l2RegularizationBlock = L2Regularization(0.1, b)
    val l2RegularizationBlockMatr = L2Regularization(0.001, matrix)
    println("\nGradient Checking L2 REGULARIZATION block wrt vectorParam: ")
    GradientChecker(l2RegularizationBlock, b)
    println("\nGradient Checking L2 REGULARIZATION block wrt matrixParam: ")
    GradientChecker(l2RegularizationBlockMatr, matrix)

    // SUM OF WORDS MODEL
    val sumOfWordsModel = new SumOfWordVectorsModel(4, 0.1)
    val score = sumOfWordsModel.scoreSentence(a)
    val loss = new LossSum(NegativeLogLikelihoodLoss(score, 1.0), sumOfWordsModel.regularizer(Seq(a,b)))
    println("\nGradient Checking SUM OF WORDS model: ")
    GradientChecker(loss, b)

    // RNN MODEL
    val rnnModel = new RecurrentNeuralNetworkModel(4, 4, 0.000001, 0.00001)
    val rnnSentence = rnnModel.wordVectorsToSentenceVector(Seq(a,b))
    val rnnScore = rnnModel.scoreSentence(rnnSentence)
    val rnnLoss = new LossSum(NegativeLogLikelihoodLoss(rnnScore, 1.0), rnnModel.regularizer(Seq(a,b)))
    println("\nGradient Checking RNN model: ")
    GradientChecker(rnnLoss, b)
}