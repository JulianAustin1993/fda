package fda
package basis

import breeze.linalg.{DenseMatrix, DenseVector, linspace}
import fda.math.integration.rombergIntegrator
import spire.implicits.cfor

trait Basis {
  /*
  Domain of the basis system.
   */
  val domain: (Double, Double)

  /*
  Number of components of the basis system.
   */
  val nComponents: Int

  /*
  Function creating a basis vector from location and order.
   */
  def design(x: Double, q: Int): DenseVector[Double]

  def designMatrix(x: DenseVector[Double], q: Int): DenseMatrix[Double] = {
    val a = x.map(xx => design(xx, q))
    val n = x.length
    val Phi = DenseMatrix.zeros[Double](n, nComponents)
    cfor(0)(_ < n, _ + 1){
      i => Phi(i, ::) := a(i).t
    }
    Phi
  }
  def penaltyMatrix(q1: Int, k: Int): DenseMatrix[Double] = {
    val J = scala.math.pow(2.0, k)
    val dx = (domain._2 - domain._1) / J
    val x = linspace(domain._1, domain._2, (J+1).toInt)
    val Phi = designMatrix(x, q1)
    val I = DenseMatrix.zeros[Double](nComponents, nComponents)
    cfor(0)(_ < nComponents, _ + 1){
      i => {
        cfor(0)( _ < i+1, _+1){
          j => {
            val y = Phi(::, i) *:* Phi(::, j)
            I(i,j) = rombergIntegrator(y, k, dx)
            I(j,i) = I(i, j)
          }
        }
      }
    }
    I
  }
}
