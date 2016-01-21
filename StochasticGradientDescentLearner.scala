package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {
  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, epochHook: (Int, Double) => Unit): Unit = {
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    for (i <- 0 until maxEpochs) {
      var accLoss = 0.0
      for (j <- 0 until iterations) {
        if (j % 1000 == 0) print(s"Iter $j\r")
        // θ = θ − α∇θE[J(θ)]
        val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)

        val objectiveJLoss = model.loss(sentence, target)
        accLoss += objectiveJLoss.forward()

        objectiveJLoss.backward()
        objectiveJLoss.update(learningRate) // updates parameters of the block

        //todo: update the parameters of the model and accumulate the loss
        //        ???
      }
      epochHook(i, accLoss)
    }
  }
}
