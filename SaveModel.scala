package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{File, PrintWriter}
import java.util.Calendar

import scala.io.Source
import scala.util.Random


object SaveModel {

  def writeWordVectorsFromModelToFile(bestModel:Model, maxWords:Int, wordStringFile:String, wordParamFile:String, vectorParamFile:String): Unit = {
    println("Create word vector files.")
    val actualWordWriter = new PrintWriter(new File("./data/assignment3/"+wordStringFile ))
    val wordWriter = new PrintWriter(new File("./data/assignment3/"+wordParamFile ))
    val paramWriter = new PrintWriter(new File("./data/assignment3/"+vectorParamFile ))
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

  def saveModelToFile(model:Model, fileName:String): Unit = {
    println("Save model to file.")
    val now = Calendar.getInstance().getTime.toString.replace(" ","").replace("/","")
    val actualWordWriter = new PrintWriter(new File("./data/assignment3/"+fileName ))
    var j = 0
    for ((paramName, paramBlock) <- model.vectorParams) {
      actualWordWriter.write(paramName+" ")
      actualWordWriter.write(paramBlock.param.toArray.mkString(",")+" \n")
      if (j % 1000 == 0) print(s"Params $j\r")
      j+=1
    }
    println("Done! : )")
  }

  def loadModelFromFile(fileName:String): Unit = {
    println("Load model from file.")
    LookupTable.trainableWordVectors.clear()
    var j = 0
    Source.fromFile("./data/assignment3/" + fileName).getLines().foreach(line => {
      val wordVectorSplit = line.split(" ")
      val word = wordVectorSplit(0)
      val vector = wordVectorSplit(1).split(",").map(i => i.toDouble)
      LookupTable.addTrainableWordVector(word, vec((0 until vector.length).map(i => vector(i)):_*) )
      if (j % 1000 == 0) print(s"Params $j\r")
      j+=1
    })
    println("Done! : )")
  }

  def printBestParamsFromLogFile(filename:String, parameterName:String, minimize:Boolean):Unit = {
      val r = s"(?<=$parameterName=)(.*?)\\s".r
      val startValue = if (minimize) Double.MaxValue else 0.0
      var bestParamsLine = ""
      Source.fromFile("./data/assignment3/" + filename)
        .getLines().foldLeft(startValue)((bestParam, line) => {
          r.findFirstIn(line) match {
            case Some(newBestParamString) =>
              val newBestParam = newBestParamString.toDouble
              if ((minimize && newBestParam < bestParam) || (!minimize && newBestParam > bestParam)) {
                bestParamsLine = line
                newBestParam
              } else {
                bestParam
              }
            case None => bestParam
          }
        })
      println(bestParamsLine)
  }

}
