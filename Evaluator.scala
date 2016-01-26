package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
 * @author rockt
 */
object Evaluator {
  def apply(model: Model, corpus: String): (Double,Double) = {
    var accLoss = 0.0
    val total = SentimentAnalysisCorpus.numExamples(corpus)
    var correct = 0.0
    for (i <- 0 until total) {
      val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
      val predict = model.predict(sentence)
      if (target == predict) correct = correct + 1

      val modelLoss = model.loss(sentence, target)
      accLoss += modelLoss.forward()
//      println(accLoss)
    }
    (100 * (correct / total), accLoss)
  }
}
