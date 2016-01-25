package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.util.control.Breaks

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {


  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, epochHook: (Int, Double, Model) => Unit): Unit = {
    var previousDevAcc = 0.0
    val epoch_loop = new Breaks
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    epoch_loop.breakable {
      for (i <- 0 until maxEpochs) {
        var accLoss = 0.0
        for (j <- 0 until iterations) {
          if (j % 1000 == 0) print(s"Iter $j\r")
          val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
          val modelLoss = model.loss(sentence, target)
//          println("modelLoss sim ="+model.vectorParams("param_w").dim)
          accLoss += modelLoss.forward()
          modelLoss.backward()
          modelLoss.update(learningRate) // updates parameters of the block
        }
        epochHook(i, accLoss, model)
        val modelValidationAcc = 100 * Evaluator(model, Main.validationSetName)
        // early stopping
        if (modelValidationAcc < previousDevAcc) epoch_loop.break()
        else previousDevAcc = modelValidationAcc
      }
    }
  }
}
