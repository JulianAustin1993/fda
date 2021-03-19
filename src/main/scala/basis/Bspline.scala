package fda
package basis

import breeze.linalg.{DenseMatrix, DenseVector, linspace}
import fda.basis.Bspline.{basisFuns, nonZeroToFull, wrapDersBasisFuns}

object Bspline {
  /** *
   * Find the span location of u in knot vector U
   * Implementation of Algorithm A2.1 from The NURBS Book by Piegl & Tiller.
   *
   * @param n Number of basis functions.
   * @param p Degree of Bspline functions
   * @param u Location of evaluation.
   * @param U Knot Vector
   * @return Index of knot span that contains u.
   */
  private def findSpan(n: Int, p: Int, u: Double, U: Vector[Double]): Int = u match {
    case _ if u == U(n + 1) => n - 1
    case _ => U.indexWhere(_ > u) - 1
  }


  /**
   * Evaluate the non-zero basis functions of Bspline.
   * Implementation of Algorithm A2.2 from The NURBS Book by Piegl & Tiller.
   *
   * @param u Point of evaluation.
   * @param p degree of Bspline
   * @param U Knot vector of Bspline
   * @return The Vector of nonzero Bspline functions evaluated at u and the integer offset.
   */
  private def basisFuns(u: Double, p: Int, U: Vector[Double]): (DenseVector[Double], Int) = {

    val i = findSpan(U.length - p - 1, p, u, U)

    def leftFun(j: Int): Double = j match {
      case _ if j == 0 => 0.0
      case _ => u - U(i + 1 - j)
    }

    def rightFun(j: Int): Double = j match {
      case _ if j == 0 => 0.0
      case _ => U(i + j) - u
    }

    val N = DenseVector.zeros[Double](p + 1)
    val rdel = Vector.tabulate[Double](p + 1)(rightFun)
    val ldel = Vector.tabulate[Double](p + 1)(leftFun)
    N(0) = 1.0
    for (j <- 1 to p) {
      var saved = 0.0
      for (r <- 0 until j) {
        val temp = N(r) / (rdel(r + 1) + ldel(j - r))
        N(r) = saved + rdel(r + 1) * temp
        saved = ldel(j - r) * temp
      }
      N(j) = saved
    }
    (N, i - p)
  }

  /**
   * Evaluate the nth derivative of the non-zero  basis functions of Bspline.
   * Implementation of Algorithm A2.2 from The NURBS Book by Piegl & Tiller.
   *
   * @param u Point of evaluation.
   * @param p degree of Bspline.
   * @param n derivative to take.
   * @param U Knot vector of Bspline.
   * @return The vector of the nth derivative of the nonzero Bspline functions evaluated at u and the integer offset.
   */
  private def dersBasisFuns(u: Double, p: Int, n: Int, U: Vector[Double]): (DenseVector[Double], Int) = {
    val i = findSpan(U.length - p - 1, p, u, U)

    def leftFun(j: Int): Double = j match {
      case _ if j == 0 => 0.0
      case _ => u - U(i + 1 - j)
    }

    def rightFun(j: Int): Double = j match {
      case _ if j == 0 => 0.0
      case _ => U(i + j) - u
    }

    val ndu = DenseMatrix.ones[Double](p + 1, p + 1) //working matrix
    val ders = DenseMatrix.zeros[Double](n + 1, p + 1) //out matrix
    val a = DenseMatrix.ones[Double](2, p + 1) // working matrix.

    val rdel = Vector.tabulate[Double](p + 1)(rightFun)
    val ldel = Vector.tabulate[Double](p + 1)(leftFun)
    for (j <- 1 to p) {
      var saved = 0.0
      for (r <- 0 until j) {
        // Lower triangle
        ndu(j, r) = rdel(r + 1) + ldel(j - r)
        val temp = ndu(r, j - 1) / ndu(j, r)

        //upper triangle
        ndu(r, j) = saved + rdel(r + 1) * temp
        saved = ldel(j - r) * temp
      }
      ndu(j, j) = saved
    }
    ders(0, ::) := ndu(::, p).t
    // Compute the derivatives
    for (r <- 0 to p) { //loop over function index
      var s1 = 0 //switching between 0, 1 each iteration
      var s2 = 1 //switching between 1, 0 each iteration
      a(0, 0) = 1.0
      for (k <- 1 to n) { //loop for kth derivative.
        var d = 0.0
        val rk = r - k
        val pk = p - k
        if (r >= k) {
          a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk)
          d = a(s2, 0) * ndu(rk, pk)
        }
        val j1 = if (rk >= -1) 1 else -rk
        val j2 = if (r - 1 <= pk) k - 1 else p - r
        for (j <- j1 to j2) {
          a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j)
          d += a(s2, j) * ndu(rk + j, pk)
        }
        if (r <= pk) {
          a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r)
          d += a(s2, k) * ndu(r, pk)
        }
        ders(k, r) = d
        val j = s1
        s1 = s2
        s2 = j
      }
    }
    //multiply through by correct factors.
    var r: Double = p
    for (k <- 1 to n) {
      for (j <- 0 to p) {
        ders(k, j) *= r
      }
      r *= (p - k)
    }
    (ders(n, ::).t, i - p)
  }


  /**
   * Wrapper to calling derBasisFuns, adds a simple check if u is on the upper boundary and
   * derivative is equal to order, in which case we can return vector of zeros.
   *
   * @param u Point of evaluation.
   * @param p degree of Bspline.
   * @param n derivative to take.
   * @param U Knot vector of Bspline.
   * @return The vector of the nth derivative of the nonzero Bspline functions evaluated at u and the integer offset.
   */
  private def wrapDersBasisFuns(u: Double, p: Int, n: Int, U: Vector[Double]): (DenseVector[Double], Int) = {
    if (u >= U.last && n == p) {
      val i = findSpan(U.length - p - 1, p, u, U)
      (DenseVector.zeros[Double](p + 1), i - p)
    } else {
      dersBasisFuns(u, p, n, U)
    }
  }

  /**
   * Create full basis vector from non-zero elements only.
   *
   * @param N        DenseVector containing non-zero basis values.
   * @param offset   Integer giving the index of start in full basis vector.
   * @param numBasis Total length of full basis vector.
   * @return
   */
  private def nonZeroToFull(N: DenseVector[Double], offset: Int, numBasis: Int): DenseVector[Double] = {
    val out = DenseVector.zeros[Double](numBasis)
    out(offset until offset + N.length) := N
    out
  }

  /**
   * Calculate the default knot placement for spline with numBasis
   * basis functions.
   *
   * @param l        lower bound of domain for knot placements.
   * @param u        upper bound of domain for knot placements.
   * @param order    Order of the Bspline function to ensure correct padding of internal knots.
   * @param numBasis Number of basis functions for the spline function.
   * @return
   */
  private def defaultKnots(l: Double, u: Double, order: Int, numBasis: Int): Vector[Double] = {
    val interior: List[Double] = linspace(l, u, numBasis - order + 2).toScalaVector.toList
    (List.fill(order - 1)(l) ::: interior ::: List.fill(order - 1)(u)).toVector
  }

  /**
   * Constructor for Bspline class with fully specified knotVector.
   * @param domain Domain range of Bspline with lower and upper bound as each element of the tuple.
   * @param nComponents Number of basis function in Bspline basis system.
   * @param splineOrder Order of the Spline functions.
   * @param knotVector The full knot vector of the Bspline basis function.
   * @return Bspline instance
   */
  def apply(domain: (Double, Double),
            nComponents: Int,
            splineOrder: Int,
            knotVector: Vector[Double]): Bspline = {
    new Bspline(d = domain, k = nComponents, o = splineOrder, kV = knotVector)
  }

  /**
   * Constructor for Natural Bspline with uniform knots over the domain.
   * @param domain Domain range of Bspline with lower and upper bound as each element of the tuple.
   * @param nComponents Number of basis function in Bspline basis system.
   * @param splineOrder Order of the Spline functions.
   * @return Bspline Instance
   */
  def apply(domain: (Double, Double),
            nComponents: Int,
            splineOrder: Int): Bspline = {
    new Bspline(d = domain,
      k = nComponents,
      o = splineOrder,
      kV = defaultKnots(domain._1, domain._2, splineOrder, nComponents))
  }
}

/**
 * Construct Bspline basis system.
 * @param d Lower and upper bounds of the domain of the system.
 * @param k Number of components in the system.
 * @param o The order of the Bspline function.
 * @param kV The knot vector of the Bspline basis system.
 */
class Bspline(d: (Double, Double),
              k: Int,
              o: Int,
              kV: Vector[Double]) extends Basis{

  override val domain: (Double, Double) = d
  override val nComponents: Int = k
  val splineOrder: Int = o
  val knotVector: Vector[Double] = kV

  /**
   * Evaluate the qth derivative of the Basis system at point x.
   *
   * @param x Point of evaluation for the basis system.
   * @param q The derivative to take. Must be greater than or equal to  zero.
   * @return The qth derivative of the basis system evaluated at x.
   */
  override def design(x: Double, q: Int): DenseVector[Double] = {
    require(q >= 0)
    require(x >= domain._1 && x <= domain._2)
    val (nonzero, offset) = q match {
      case _ if q == 0 => basisFuns(x, splineOrder - 1, knotVector)
      case _ if q > 0 => wrapDersBasisFuns(x, splineOrder - 1, q, knotVector)
    }
    nonZeroToFull(nonzero, offset, nComponents)
  }

}
