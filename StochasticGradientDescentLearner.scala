package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.util.control.Breaks

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {

  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double): Unit = {
    var previousDevLoss = Double.MaxValue
    val epoch_loop = new Breaks
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    epoch_loop.breakable {
      for (i <- 0 until maxEpochs) {
        var accLoss = 0.0
        for (j <- 0 until iterations) {
          if (j % 1000 == 0) print(s"Iter $j\r")
          val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
          val modelLoss = model.loss(sentence, target)
          accLoss += modelLoss.forward()
          modelLoss.backward()
          modelLoss.update(learningRate) // updates parameters of the block
        }
//        if (i % 3 == 0) {
          previousDevLoss = epochHook(i, accLoss, model, epoch_loop, previousDevLoss)
//        }
      }
    }
  }

  def epochHook(iter: Int, accLoss: Double, model: Model, epoch_loop:Breaks, previousDevLoss:Double): Double = {
    val evaluatorOnTrainSet = Evaluator(model, Main.trainSetName) // accuracy percentage, loss
    val evaluatorOnValidSet = Evaluator(model, Main.validationSetName)
    println("Epoch %4d\tLoss %8.2f\tTrain_Acc %4.2f\tDev_Acc %4.2f\t%4.2f\n".format(
      iter, accLoss, evaluatorOnTrainSet._1, evaluatorOnValidSet._1, evaluatorOnValidSet._2))

    // early stopping if loss on valid. set not going down
    if (evaluatorOnValidSet._2 > previousDevLoss || evaluatorOnTrainSet._2.isNaN) {
      epoch_loop.break()
      println("Break! " + evaluatorOnValidSet._2)
    } else {
      println(evaluatorOnValidSet._2)
    }
    evaluatorOnValidSet._2
  }

}
