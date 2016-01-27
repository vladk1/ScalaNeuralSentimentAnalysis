package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.{File, PrintWriter}

import scala.util.Random


object SaveModel {

  def writeWordVectorsFromModelToFile(bestModel:Model, maxWords:Int): Unit = {
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


}
