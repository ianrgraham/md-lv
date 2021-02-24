use num_traits::Float;

/// An iterator of a sequence of evenly spaced floats.
///
/// Iterator element type is `F`.
#[derive(Clone, Debug)]
pub struct Linspace<F> {
    start: F,
    step: F,
    index: usize,
    len: usize,
}

impl<F> Iterator for Linspace<F>
    where F: Float
{
    type Item = F;

    #[inline]
    fn next(&mut self) -> Option<F> {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            let i = self.index;
            self.index += 1;
            Some(self.start + self.step * F::from(i).unwrap())
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;
        (n, Some(n))
    }
}

impl<F> DoubleEndedIterator for Linspace<F>
    where F: Float,
{
    #[inline]
    fn next_back(&mut self) -> Option<F> {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            self.len -= 1;
            let i = self.len;
            Some(self.start + self.step * F::from(i).unwrap())
        }
    }
}

impl<F> ExactSizeIterator for Linspace<F>
    where Linspace<F>: Iterator
{}

/// Return an iterator of evenly spaced floats.
///
/// The `Linspace` has `n` elements, where the first
/// element is `a` and the last element is `b`.
///
/// Iterator element type is `F`, where `F` must be
/// either `f32` or `f64`.
///
/// ```
/// extern crate itertools;
/// extern crate itertools_num;
///
/// use itertools_num::linspace;
///
/// # fn main() {
/// itertools::assert_equal(linspace::<f32>(0., 1., 5),
///                         vec![0., 0.25, 0.5, 0.75, 1.0]);
/// # }
/// ```
#[inline]
pub fn linspace<F>(a: F, b: F, n: usize) -> Linspace<F>
    where F: Float
{
    let step = if n > 1 {
        let nf: F = F::from(n).unwrap();
        (b - a) / (nf - F::one())
    } else {
        F::zero()
    };
    Linspace {
        start: a,
        step: step,
        index: 0,
        len: n,
    }
}
