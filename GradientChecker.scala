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
//  Question 2

//    initialise vectors a and b
  val a = vec(-1.5, 1.0, 1.5, 0.5)
  val b = VectorParam(4)
  b.set(vec(1.0, 2.0, -0.5, 2.5))

//  dot block gradient check
  val simpleDotBlock = Dot(a, b)
  print("SimpleDotBlock wrt b")
  GradientChecker(simpleDotBlock, b)

//  sum block
  val sumBlock = Sum(Seq(a, b))
  print("SumBlock wrt b")
  GradientChecker(Dot(sumBlock, sumBlock), b)


// sigmoid block
  val sigmoidBlock = Sigmoid(Dot(a, b))
  print("SigmoidBlock wrt b")
  GradientChecker(sigmoidBlock, b)

//  NLL block
  val negativeLogLikelihoodLossBlock = NegativeLogLikelihoodLoss(Sigmoid(Dot(a, b)), 0.5)
  print("NegativeLogLikelihoodLossBlock wrt b")
  GradientChecker(negativeLogLikelihoodLossBlock, b)

// Sum of Words block
  val sumOfWordsModel = new SumOfWordVectorsModel(4, 0.1)
  print("SumOfWordsModel wrt b")
  val score = sumOfWordsModel.scoreSentence(a)
  val loss = new LossSum(NegativeLogLikelihoodLoss(score, 1.0), sumOfWordsModel.regularizer(Seq(a,b)))
  GradientChecker(loss, b)

//  Question 3

//  initialise matrix
  val matrix = MatrixParam(4,4)
//  matrix.set(mat(2,2)(-1.5, 1.0, -2.0, 2.2))

// mul block
//  val mulBlock = Dot(Mul(matrix, a), b)
//  println(mulBlock.forward())
  println("mul block wrt b")
  GradientChecker(Dot(Mul(matrix, a), b), b)

//  tanh block
  val simpleTanBlock = Tanh(b)
  println("dot(tan) wrt b: ")
  GradientChecker(Dot(simpleTanBlock, simpleTanBlock), b)


// rnn block
  val rnnModel = new RecurrentNeuralNetworkModel(4, 4, 0.01, 0.0001)
  val rnnSentence = rnnModel.wordVectorsToSentenceVector(Seq(a,b))
  val rnnScore = rnnModel.scoreSentence(rnnSentence)
  val rnnLoss = new LossSum(NegativeLogLikelihoodLoss(rnnScore, 1.0), rnnModel.regularizer(Seq(a,b)))
  print("rnnModel wrt b: ")
  GradientChecker(rnnLoss, b)

// elementMul Block
  val simpleElementMulBlock = Dot(ElementMul(a,a),b)
  print("elementMul wrt b: ")
  GradientChecker(simpleElementMulBlock, b)

// vectorSigmoid Block
  val simpleVectorSigmoidBlock = Dot(VectorSigmoid(a), b)
  print("vectorSigmoid wrt b: ")
  GradientChecker(simpleVectorSigmoidBlock, b)

}
