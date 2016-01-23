package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {
  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, epochHook: (Int, Double, Model) => Unit): Unit = {
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    for (i <- 0 until maxEpochs) {
      var accLoss = 0.0
      for (j <- 0 until iterations) {
        if (j % 1000 == 0) print(s"Iter $j\r")

        // θ = θ − α∇θE[J(θ)]
        val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)

        val modelLoss = model.loss(sentence, target)
        accLoss += modelLoss.forward()
        modelLoss.backward()
        modelLoss.update(learningRate) // updates parameters of the block
      }
      epochHook(i, accLoss, model)
    }
  }
}
