package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg._
import breeze.numerics.{tanh, log, pow, sigmoid}

import scala.util.Random


/**
  * @author rockt
  */

/**
  * A trait for the core building **block** of our computation graphs
  *
  * @tparam T the type parameter this block evaluates to (can be Double, Vector, Matrix)
  */
trait Block[T] {
  //caches output after call of forward
  var output: T = _
  //fun fact: when you say "forward" or "back" your lips move in the respective direction
  def forward(): T
  //assumes that forward has been called first!
  def backward(gradient: T): Unit
  //updates parameters of the block
  def update(learningRate: Double)
}

/**
  * A loss function is a block that evaluates to a single double.
  * Generally loss functions don't have upstream gradients,
  * so we can provide an implementation of backward without arguments.
  */
trait Loss extends Block[Double] {
  def backward(): Unit
}

trait ParamBlock[P] extends Block[P] {
  var param: P
  val gradParam: P
  def initialize(dist: () => Double): P
  def set(p: P): Unit = {
    param = p
  }
  def resetGradient(): Unit
}

trait DefaultInitialization {
  def defaultInitialization(): Double
}

/**
  * This trait defines a default way of initializing weights of parameters
  */
trait GaussianDefaultInitialization extends DefaultInitialization {
  def defaultInitialization(): Double = random.nextGaussian() * 0.1 // [-0.1, 0.1]
}

/**
  * A simple block that represents a constant double value
  *
  * @param arg the constant double value
  */
case class DoubleConstant(arg: Double) extends Block[Double] with Loss {
  output = arg
  def forward(): Double = output
  def backward(gradient: Double): Unit = {} //nothing to do since fixed
  def update(learningRate: Double): Unit = {} //nothing to do since fixed and no child blocks
  def backward(): Unit = {} //nothing to do since fixed
}

/**
  * A simple block that represents a constant vector
  *
  * @param arg the constant vector
  */
case class VectorConstant(arg: Vector) extends Block[Vector] {
  output = arg
  def forward(): Vector = output
  def backward(gradient: Vector): Unit = {} //nothing to do since fixed
  def update(learningRate: Double): Unit = {} //nothing to do since fixed and no child blocks
}

/**
  * A block representing a sum of doubles
  *
  * @param args a sequence of blocks that evaluate to doubles
  */
case class DoubleSum(args: Block[Double]*) extends Block[Double] {
  def forward(): Double = {
    output = args.map(_.forward()).sum
    output
  }
  def backward(gradient: Double): Unit = args.foreach(_.backward(gradient))
  def update(learningRate: Double): Unit = args.foreach(_.update(learningRate))
}

class LossSum(override val args: Loss*) extends DoubleSum(args:_*) with Loss {
  def backward(): Unit = args.foreach(_.backward())
}



/**
  * Problem 2
  */

/**
  * A block representing a vector parameter
  *
  * @param dim dimension of the vector
  * @param clip defines range in which gradients are clipped, i.e., (-clip, clip)
  */
case class  VectorParam(dim: Int, clip: Double = 10.0) extends ParamBlock[Vector] with GaussianDefaultInitialization {
  var param: Vector = initialize(defaultInitialization) // sigmoidInitialization
  val gradParam: Vector = DenseVector.zeros[Double](dim)

  /**
    * @return the current value of the vector parameter and caches it into output
    */
  def forward(): Vector = {
    output = param
    output
  }
  /**
    * Accumulates the gradient in gradParam
    *
    * @param gradient an upstream gradient
    */
  def backward(gradient: Vector): Unit = gradParam :+= gradient
  /**
    * Resets gradParam to zero
    */
  def resetGradient(): Unit = {
//    for(i <- 0 until gradParam.activeSize)
      gradParam := DenseVector.zeros[Double](dim)
  }
  /**
    * Updates param using the accumulated gradient. Clips the gradient to the interval (-clip, clip) before the update
    *
    * @param learningRate learning rate used for the update
    */
  def update(learningRate: Double): Unit = {
    param :-= (breeze.linalg.clip(gradParam, -clip, clip) * learningRate) //in-place
    resetGradient()
  }
  /**
    * Initializes the parameter randomly using a sampling function
    *
    * @param dist sampling function
    * @return the random parameter vector
    */
  def initialize(dist: () => Double): Vector = {
    param = randVec(dim, dist)
    param
  }
}

/**
  * A block representing the sum of vectors
  *
  * @param args a sequence of blocks that evaluate to vectors
  */
case class Sum(args: Seq[Block[Vector]]) extends Block[Vector] {
  def forward(): Vector = {
    val stepSumVector = args.map(_.forward())
    val init = DenseVector.zeros[Double](stepSumVector(0).output.activeSize)
    output = stepSumVector.foldRight(init)((arg, sum) => {
      sum :+= arg
    })
    output
  }
  def backward(gradient: Vector): Unit = {
    args.foreach(_.backward(gradient))
  }
  def update(learningRate: Double): Unit = args.foreach(_.update(learningRate))
}

case class Sub(args: Seq[Block[Vector]]) extends Block[Vector] {
  def forward(): Vector = {
    val stepSumVector = args.map(_.forward())
    val init = vec((0 until stepSumVector(0).activeSize).map(i => 0.0):_*)
    output = stepSumVector.foldRight(init)((arg, sum) => {
      sum :-= arg
    })
    output
  }
  def backward(gradient: Vector): Unit = {
    args.foreach(_.backward(gradient))
  }
  def update(learningRate: Double): Unit = args.foreach(_.update(learningRate))
}

/**
  * A block representing the dot product between two vectors
  *
  * @param arg1 left block that evaluates to a vector
  * @param arg2 right block that evaluates to a vector
  */
case class Dot(arg1: Block[Vector], arg2: Block[Vector]) extends Block[Double] {
  def forward(): Double = {
    output = arg1.forward() dot arg2.forward()
    output
  }

  def backward(gradient: Double): Unit = {
    arg1.backward(arg2.output * gradient)
    arg2.backward(arg1.output * gradient)
  }

  def update(learningRate: Double): Unit = {
    arg1.update(learningRate)
    arg2.update(learningRate)
  }
}

/**
  * A block representing the sigmoid of a scalar value
  *
  * @param arg a block that evaluates to a double
  */
case class Sigmoid(arg: Block[Double]) extends Block[Double] {
  def forward(): Double = {
    output = sigmoid(arg.forward())
    output
  }
  def backward(gradient: Double): Unit = {
    val x = arg.output
    val localGradient = sigmoid(x) * (1 - sigmoid(x))
    arg.backward(localGradient * gradient)
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
}

/**
  * A block representing the negative log-likelihood loss
  *
  * @param arg a block evaluating to a scalar value
  * @param target the target value (1.0 positive sentiment, 0.0 negative sentiment)
  */
case class NegativeLogLikelihoodLoss(arg: Block[Double], target: Double) extends Loss {
  def forward(): Double = {
    val x = arg.forward()
    output = - target * log(x) - (1-target) * log(1-x)
    output
  }
  // loss functions are root nodes so they don't have upstream gradients
  def backward(gradient: Double): Unit = backward()
  def backward(): Unit = {
    val x = arg.output
    val gradient = - target/x + (1 - target)/(1 - x)
    arg.backward(gradient)
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
}

/**
  * A block representing the l2 regularization of a vector or matrix
  *
  * @param strength the strength of the regularization (often denoted as lambda)
  * @param args a block evaluating to a vector or matrix
  * @tparam P type of the input block (we assume this is Block[Vector] or Block[Matrix]
  */
case class L2Regularization[P](strength: Double, args: Block[P]*) extends Loss {
  def forward(): Double = {
    /**
      * Calculates the loss individually for every vector/matrix parameter in args
      */
    // Vlad: R(x) = strength * 0.5 * sum(x_1^2 + x_2^2 + x_3^2...)
    val losses = args.map(arg => {
      val in = arg.forward()
      in match {
        case v: Vector => strength * 0.5 * sum(v.map(x => pow(x,2)))
        case w: Matrix => strength * 0.5 * sum(w.toDenseVector.map(x => pow(x,2)))
      }
    })
    output = sum(losses) //sums the losses up
    output
  }
  def update(learningRate: Double): Unit = args.foreach(_.update(learningRate))
  //loss functions are root nodes so they don't have upstream gradients
  def backward(gradient: Double): Unit = backward()
  // Vlad: R(x)' = vec(strength*x_1, strength*x_2, strength*x_3 ...)
  def backward(): Unit = args.foreach(x => x.backward((x.output match {
    case v: Vector => v :* strength
//    case w: Matrix => w.toDenseVector.map(w_i => strength * w_i)
    case w: Matrix => w :* strength
  }).asInstanceOf[P]))
}



/**
  * Problem 3
  */

/**
  * A block representing a matrix parameter
  *
  * @param dim1 first dimension of the matrix
  * @param dim2 second dimension of the matrix
  * @param clip defines range in which gradients are clipped, i.e., (-clip, clip)
  */
case class MatrixParam(dim1: Int, dim2: Int, clip: Double = 10.0) extends ParamBlock[Matrix] with GaussianDefaultInitialization {
  var param: Matrix =  initialize(defaultInitialization) // sigmoidInitialization defaultInitialization tanhInitialisation
  val gradParam: Matrix = DenseMatrix.zeros[Double](dim1,dim2)
  def forward(): Matrix = {
    output = param
    output
  }
  def backward(gradient: Matrix): Unit = gradParam :+= gradient
  def resetGradient(): Unit = {
    for(i <- 0 until dim1)
      for(j <- 0 until dim2){
        gradParam(i,j) = 0.0
      }
  }
  def update(learningRate: Double): Unit = {
    param :-= (breeze.linalg.clip(gradParam, -clip, clip) * learningRate)
    resetGradient()
  }
  def initialize(dist: () => Double): Matrix = {
    param = randMat(dim1, dim2, dist)
    param
  }
}

/**
  * A block representing matrix-vector multiplication
  *
  * @param arg1 the left block evaluating to a matrix
  * @param arg2 the right block evaluation to a vector
  */
case class Mul(arg1: Block[Matrix], arg2: Block[Vector]) extends Block[Vector] {
  def forward(): Vector = arg1.forward() * arg2.forward()
  def backward(gradient: Vector): Unit = {
    arg1.backward(outer(gradient, arg2.output)) // (original) gradient *  arg2'
    arg2.backward(arg1.output.t * gradient) // arg1' * gradient
  }
  def update(learningRate: Double): Unit = {
    arg1.update(learningRate)
    arg2.update(learningRate)
  }
}

/**
  * A block rerpesenting the element-wise application of the tanh function to a vector
  *
  * @param arg a block evaluating to a vector
  */
case class Tanh(arg: Block[Vector]) extends Block[Vector] {
  def forward(): Vector = {
    output = tanh(arg.forward())
    output
  }
  def backward(gradient: Vector): Unit = {
    val in = arg.output
    val local = {
      in.map(x => 1 - pow(tanh(x), 2))
    }
    arg.backward(gradient :* local)
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
}



/**
  * Problem 4
  */

/**
  * A potentially useful block for training a better model (https://en.wikipedia.org/wiki/Dropout_(neural_networks))
  *
  * @param prob dropout probability
  * @param arg a block evaluating to a vector whose components we want to drop
  */
case class Dropout(prob: Double, arg: Block[Vector]) extends Block[Vector] {
  val isDrop = {
    val x = Random.nextInt(100)
    x < 100*prob
  }

  def forward(): Vector = {
    val argument = arg.forward()
    output = DenseVector.zeros[Double](argument.activeSize)

    if (isDrop) {
      output = argument
    }
    output
  }

  def update(learningRate: Double): Unit = {
    arg.update(learningRate)
  }
  def backward(gradient: Vector): Unit = {
    arg.backward(gradient)
  }
}


case class DropoutForMul(prob: Double, arg: Block[Vector]) extends Block[Vector] {
  val isDrop = {
    val x = Random.nextInt(100)
    x < 100*prob
  }

  def forward(): Vector = {
    val argument = arg.forward()
    output = DenseVector.zeros[Double](argument.activeSize)

    if (isDrop) {
      output = argument
    }
    output
  }

  def update(learningRate: Double): Unit = {
    arg.update(learningRate)
  }
  def backward(gradient: Vector): Unit = {
    arg.backward(gradient)
  }
}

/**
  * A block representing the sigmoid of a vector value
  *
  * @param arg a vector that evaluates to a vector
  */
case class VectorSigmoid(arg: Block[Vector]) extends Block[Vector] {
  def forward(): Vector = {
    output = sigmoid(arg.forward())
    output
  }
  def backward(gradient: Vector): Unit = {
      val in = arg.output
      val local = {
       in.map(x => (sigmoid(x) * (1 - sigmoid(x))))
      }
      arg.backward(gradient :* local)
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
}

/**
  * A block representing element wise multiplication
  *
  * @param args the sequence of blocks involved in element wise multiplication
  */

case class ElementMul(args: Seq[Block[Vector]]) extends Block[Vector]{

  def forward(): Vector = {
    val mulVect = args.map(arg => arg.forward())

    var init = DenseVector.zeros[Double](mulVect(0).output.activeSize) :+ 1.0

    output = mulVect.foldRight(init)((arg, prod) => {
      prod :* arg
    })
    output
  }

  def backward(gradient: Vector): Unit = {

    var curGrad = output
    curGrad = curGrad :* gradient

    args.foreach(arg => {
      arg.backward(curGrad :/ arg.output)
    })
  }

  def update(learningRate: Double): Unit = {
    args.foreach(_.update(learningRate))
  }
}

//case class ElementMul(arg1: Block[Vector], arg2: Block[Vector]) extends Block[Vector] {
//
//  def forward(): Vector = {
//    arg1.forward() :* arg2.forward()
//  }
//
//  def backward(gradient: Vector): Unit = {
//    arg1.backward(gradient :* arg2.output)
//    arg2.backward(gradient :* arg1.output)
//  }
//
//  def update(learningRate: Double): Unit = {
//    arg1.update(learningRate)
//    arg2.update(learningRate)
//  }
//}