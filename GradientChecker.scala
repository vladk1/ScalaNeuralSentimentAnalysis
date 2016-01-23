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
  val a = vec(-1.5, 1.0, 1.5, 0.5)
  val b = VectorParam(4)
  println("b" + b.forward())
  b.set(vec(1.0, 2.0, -0.5, 2.5))
  println("b after" + b.forward())
  val simpleBlock = Dot(a, b)
  GradientChecker(simpleBlock, b)

  val matrix = MatrixParam(2,2)
  matrix.set(mat(2,2)(-1.5, 1.0, -2.0, 2.2))
  val matr1 = MatrixParam(3,3)
//  matr1.set(mat(3,3)(0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 0.0, 0.0))
  val vect = VectorParam(3)
//  vect.set(vec(1.0, 2.0, 5.0))

  b.set(vec(1.0, 2.0, -0.5, 2.5))
  val simpleDotBlock = Dot(a, b)
  GradientChecker(simpleDotBlock, b)

  val sumBlock = Sum(Seq(a,b))
  GradientChecker(Dot(sumBlock, sumBlock), b)

  val sigmoidBlock = Sigmoid(Dot(a, b))
  GradientChecker(sigmoidBlock, b)

  val negativeLogLikelihoodLossBlock = NegativeLogLikelihoodLoss(Sigmoid(Dot(a, b)), 0.5)
  GradientChecker(negativeLogLikelihoodLossBlock, b)

  val l2RegularizationBlock = L2Regularization(1, b)
  val l2RegularizationBlockMatr = L2Regularization(1, matrix)

  GradientChecker(l2RegularizationBlock, b)
  GradientChecker(l2RegularizationBlockMatr, matrix)

  val mulBlock = Mul(matr1, vect)
  GradientChecker(Dot(mulBlock,mulBlock), matr1)

  val simpleTanBlock = Tanh(b)
  GradientChecker(Dot(simpleTanBlock, simpleTanBlock), b)

  // test RNN model
  val w = VectorParam(15)
  val h0 = VectorParam(10)
  val bias = VectorParam(10)
  val wx = MatrixParam(10, 10)
  val wh = MatrixParam(10, 10)

  val sentence = VectorParam(10)

  val Whhtprev = Mul(wh, h0)
  val Wxxt = Mul(wx, sentence)
  val score = Tanh(Sum(Seq(Whhtprev, Wxxt, bias)))
  val RNNBlock = Sigmoid(Dot(score, bias))
  println("Gradient check RNN block")
  GradientChecker(RNNBlock, bias)

}
