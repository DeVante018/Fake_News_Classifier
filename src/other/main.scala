package other
//this file contains the function used to split the tsv file into classifications and clean them
import java.io.{File, PrintWriter}
import scala.collection.mutable
import scala.io.Source

object main {
  def main(args: Array[String]): Unit = {
    sortClassify()
  }

  def getListOfFiles(dir: String): List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  def cleanData(): Unit = {
    val listOfPngFiles: List[File] = getListOfFiles("public_image_set")
    val mapOfImages: mutable.Map[String, Boolean] = mutable.Map()
    val newFile = new PrintWriter(new File("multimodal_validate_clean.tsv"))
    for (item <- listOfPngFiles) {
      mapOfImages.+=((item.toString, true)) //put name of the .png in the map
    }
    val bufferedSource = Source.fromFile("src/multimodal_validate.tsv")
    for (line <- bufferedSource.getLines.toList) {
      val tabs = line.split("\t").toList
      //println(tabs(5))//should be the image id
      if (tabs.size >= 6) {
        var imageId = "public_image_set/"
        imageId += tabs(5)
        imageId += ".jpg"
        //if(mapOfImages(imageId)){
        //newFile.write(line.toString)
        //}
        var boolis: Boolean = false
        for (keys <- mapOfImages) {
          //println(keys._1,imageId)
          if (keys._1 == imageId) {
            boolis = true
          }
        }
        if (boolis) {
          newFile.write(line)
          newFile.write("\n")
        }
      }

    }
    newFile.close()
    bufferedSource.close
  }

  def sortClassify(): Unit = {
    val zero = new PrintWriter(new File("multimodal_test_public_0"))
    val one = new PrintWriter(new File("multimodal_test_public_1"))
    val two = new PrintWriter(new File("multimodal_test_public_2"))
    val bufferedSource = Source.fromFile("src/clean/multimodal_test_public_clean.tsv")
    for (line <- bufferedSource.getLines.toList) {
      val tabs = line.split("\t").toList
      //println("3 way: ", tabs(14))//
      if (tabs.size > 14) {
        if (tabs(14).toInt == 0) {
          zero.write(line)
          zero.write("\n")
        }
        else if (tabs(14).toInt == 1) {
          one.write(line)
          one.write("\n")
        }
        else if (tabs(14).toInt == 2) {
          two.write(line)
          two.write("\n")
        }
        else {

        }
      }
    }
    zero.close()
    one.close()
    two.close()
    bufferedSource.close
  }
}
