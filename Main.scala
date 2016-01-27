package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{File, PrintWriter}
import scala.collection.mutable
import scala.util.Random
import scala.util.control.Breaks

/**
 * @author rockt
 */
/**
 * Example training of a model
 *
 * Problems 2/3/4: perform a grid search over the parameters below
 */
object Main extends App {
  /**
   * Example training of a model
   *
   * Problems 2/3/4: perform a grid search over the parameters below
   */

//    Using for RNN (single run)
  val learningRate = 0.01 // tried 0.01, 0.1, 1.0, 10.0, 1.0,
  val vectorRegularizationStrength = 0.001 // tried 0.01, 0.1, 0.01, _, _, 5.0
  val matrixRegularizationStrength = 0.001 // tried 0.01, 0.1, 0.01, _, _, 5.0, 0.01
  val wordDim = 1 // tried 10, 5
  val hiddenDim = 1 // tried 10, 5

  val trainSetName = "debug"
  val validationSetName = "debug"

  val wordDimSet = 10 to 11 by 1
  val vectorRegStrengthSet = (-6.0 to 0.0 by 1.0).map(a => Math.pow(10,a)) // in case Nan - higher regularizer
  // in case Nan - lower learning rate (hence we can stop iterating after reaching Nan in this loop)
  val learningRateSet = (-6.0 to 0.0 by 0.5).map(a => Math.pow(10,a))


  //11) 75.37 (7,1.0E-5,0.1)1 iter => deeper
//  10,  0.1,  0.0031622776601683794
//  10, 0.01, 0.01, 83.65680110666567, 76.36484687083887
//  val testModel = new SumOfWordVectorsModel(10, 0.01)
//  StochasticGradientDescentLearner(testModel, trainSetName, 100, 0.01)

// q. 4.3.4)
//    runGridSearch(wordDimSet, vectorRegStrengthSet, learningRateSet, 100)
  //  ToDo we can visualize parameter space i.e. 3d graph of params and validation set

// q. 4.3.5)
  //  iterateThroughEpochs(wordDim:Int, vectorRegularizationStrength:Double, learningRate:Double)

// q. 4.3.6) prints vectorparams to file to visualize them later
//    writeVectorsFromBestModelToFile(10, 0.01, 0.01, 100, 500)



// run norm SGDL with RNN model for debug
// q. 4.4.2)
//  val rnnModel = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
//  StochasticGradientDescentLearner(rnnModel, trainSetName, 100, learningRate)
  val LSTMModel = new LSTMModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
  StochasticGradientDescentLearner(LSTMModel, trainSetName, 100, learningRate)

  def runGridSearch(wordDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {
    val historyWriter = new PrintWriter(new File("./data/assignment3/param_history.txt" ))
    val gridSearchParams = mutable.MutableList[(Int, Double, Double, Double, Double)]()
    val loop = new Breaks

    for (wordDim <- wordDimSet; vectorRegStrength <- vectorRegStrengthSet) {
        loop.breakable {
          for (learningRate <- learningRateSet) {
            println(LookupTable.trainableWordVectors.size)
            LookupTable.trainableWordVectors.clear()
            println(LookupTable.trainableWordVectors.size)
            runSGDwithParam(wordDim, vectorRegStrength, learningRate, epochs)
          }
      }
    }
    historyWriter.close()
    if (gridSearchParams.isEmpty) {
      println("No best grid search parameters at that range!")
    } else {
      val bestDevAcc = gridSearchParams.maxBy(param => param._5)
      println("bestDevAcc=" + bestDevAcc)
    }

    def runSGDwithParam(wordDim:Int, vectorRegStrength:Double, learningRate:Double, epochs:Int):Unit = {
      val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegStrength)
      StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate)

      println("wordDim %d\tvectorRegStrength %4.10f\tlearningRate %4.10f\t".format(wordDim, vectorRegStrength, learningRate))
      if (false && hasExplodingGradient(gridSearchModel)) {
        loop.break() // don't increase learning rate, as we already have exploding gradients
        println("learningRateLoop exploded gradients")
      } else {
        val ratioOnTrainSet = Evaluator(gridSearchModel, trainSetName)._1
        val ratioOnValidSet = Evaluator(gridSearchModel, validationSetName)._1
        gridSearchParams.+=((wordDim, vectorRegStrength, learningRate, ratioOnTrainSet, ratioOnValidSet))
        println("ratioOnTrainSet %4.2f\tratioOnValidSet %4.2f\t".format(ratioOnTrainSet, ratioOnValidSet))
        historyWriter.write(wordDim + " " + vectorRegStrength + " " + learningRate + " " + ratioOnValidSet+"\n")
      }
      println("\n")
    }
}

  def hasExplodingGradient(model: Model): Boolean = {
    val (sentence, target) = SentimentAnalysisCorpus.getExample(trainSetName)
    val modelLoss = model.loss(sentence, target)
    modelLoss.forward().isNaN
  }

  def writeVectorsFromBestModelToFile(bestWordDim:Int, bestVectorRegularizationStrength:Double, bestLearningRate:Double, epochs:Int, maxWords:Int): Unit = {
    val bestModel = new SumOfWordVectorsModel(bestWordDim, bestVectorRegularizationStrength)
    StochasticGradientDescentLearner(bestModel, trainSetName, epochs, bestLearningRate)
    println("Create word vector files.")
    val actualWordWriter = new PrintWriter(new File("./data/assignment3/actual_word_param100.txt" ))
    val wordWriter = new PrintWriter(new File("./data/assignment3/word_param100.txt" ))
    val paramWriter = new PrintWriter(new File("./data/assignment3/vector_params100.txt" ))
    var count = 0

    val params: Array[(String,VectorParam)] = bestModel.vectorParams.toArray
    val rnd = new Random()

    while (count < maxWords) {
      val example = params(rnd.nextInt(params.length))
      val paramName = example._1
      val paramBlock = example._2
      println(s"$paramName:\n${paramBlock.param}\n")
      val predict = bestModel.predict(Seq(paramName))
      actualWordWriter.write(paramName+ "\n")
      wordWriter.write(predict.compare(false) + "\n")

      val wordParam = paramBlock.param
      wordParam.foreach(param => {
        paramWriter.write(param+" ")
      })
      paramWriter.write("\n")
      count += 1
    }
    wordWriter.close()
    paramWriter.close()
    actualWordWriter.close()
    println("We have written "+count+" words")
  }

  def iterateThroughEpochs(wordDim:Int, vectorRegularizationStrength:Double, learningRate:Double): Unit = {
    for (epochs <- 1 to 50 by 1) {
      val gridSearchModel = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
      StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate)
    }
  }

  /**
    * RNN Section
    */

//  75.37 (7,1.0E-5,0.1)1 iter => deeper
//  val RNNmodel: Model = new RecurrentNeuralNetworkModel(10, 10, 0.01, 0.01)
//  StochasticGradientDescentLearner(RNNmodel, trainSetName, 100, 0.01)


  /**
    * Define parameter ranges for RNN grid search
    */
  val wordDimRange = 6 to 10 by 2
//  val hiddenDimRange = 6 to 12 by 2
  val vectorRegStrengthRange = (-5.0 to -1.0 by 1.0).map(a => Math.pow(10,a)) // in case Nan - higher regularizer
  val matrixRegStrengthRange = (-4.0 to -1.0 by 1.0).map(a => Math.pow(10,a)) // in case Nan - higher regularizer
  // in case Nan - lower learning rate (hence we can stop iterating after reaching Nan in this loop)
  val learningRateRange = (-3.0 to -1.0 by 1.0).map(a => Math.pow(10,a))

//  runGridSearchRNN(wordDimRange, wordDimRange, vectorRegStrengthRange, vectorRegStrengthRange, learningRateRange, 100)

  def runGridSearchRNN(wordDimRange:Range, hiddenDimSet:Range, vectorRegStrengthSet:IndexedSeq[Double], matrixRegStrengthSet:IndexedSeq[Double],  learningRateSet:IndexedSeq[Double], epochs:Int): Unit = {
    val historyWriter = new PrintWriter(new File("./data/assignment3/param_history_rnn.txt"))
    val gridSearchParams = mutable.MutableList[(Int, Int, Double, Double, Double, Double, Double)]()
    val loop = new Breaks

    for (wordDim <- wordDimRange; vectorRegStrength <- vectorRegStrengthSet) {
      loop.breakable {
        for (learningRate <- learningRateSet) {
//          println(LookupTable.trainableWordVectors.size)
          LookupTable.trainableWordVectors.clear()
//          println(LookupTable.trainableWordVectors.size)
          runSGDwithParamRNN(wordDim, wordDim, vectorRegStrength, vectorRegStrength, learningRate, epochs)
        }
      }
    }
    historyWriter.close()
    if (gridSearchParams.isEmpty) {
      println("No best grid search parameters at that range!")
    } else {
      val bestDevAcc = gridSearchParams.maxBy(param => param._7)
      println("bestDevAcc=" + bestDevAcc)
    }


    def runSGDwithParamRNN(wordDim:Int, hiddenDim:Int, vectorRegStrength:Double, matrixRegStrength:Double, learningRate:Double, epochs:Int):Unit = {
      val gridSearchModel = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegStrength, matrixRegStrength)
      StochasticGradientDescentLearner(gridSearchModel, trainSetName, epochs, learningRate)

      println("wordDim = hiddenDim %d\tvectorRegStrength %4.10f\t vectorRegStrength %4.10f\t learningRate %4.10f\t".format(wordDim, vectorRegStrength, vectorRegStrength, learningRate))
      if (hasExplodingGradient(gridSearchModel)) {
        loop.break() // don't increase learning rate, as we already have exploding gradients
        println("learningRateLoop exploded gradients")
      } else {
          val ratioOnTrainSet = Evaluator(gridSearchModel, trainSetName)._1
          val ratioOnValidSet = Evaluator(gridSearchModel, validationSetName)._1
          gridSearchParams.+=((wordDim, hiddenDim, vectorRegStrength, matrixRegStrength, learningRate, ratioOnTrainSet, ratioOnValidSet))
          println("ratioOnTrainSet %4.2f\tratioOnValidSet %4.2f\t".format(ratioOnTrainSet, ratioOnValidSet))
          historyWriter.write(wordDim + " " + hiddenDim + " " + vectorRegStrength + " " + matrixRegStrength + " " + learningRate + " " + ratioOnValidSet + "\n")
      }
      println()
    }
  }

  /**
   * Comment this in if you want to look at trained parameters
   */
  /*
  for ((paramName, paramBlock) <- model.vectorParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  */
}