// This is a comment, and is ignored by the compiler.
// You can test this code by clicking the "Run" button over there ->
// or if you prefer to use your keyboard, you can use the "Ctrl + Enter"
// shortcut.

// This code is editable, feel free to hack it!
// You can always return to the original code by clicking the "Reset" button ->

// use std::error::Error;
// use std::{clone, fs::read_to_string};
use itertools::iproduct;
use itertools::Itertools;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::read_to_string;

// This is the main function.
fn main() {
    // Statements here are executed when the compiled binary is called.

    // Print text to the console.
    // println!("Hello World!");

    let (year, day) = (2024, 1);
    let _input = format!("https://adventofcode.com/{year}/day/{day}/input");
    // println!("Input: {}", _input);

    // let resp = httpbin();
    // println!("{:#?}", resp);

    // let resp = input_raw(day);
    // println!("{:#?}", resp);

    _day1();

    _day2();

    _day3();

    _day4();

    _day5();
}

trait Then<T> {
    fn then<F: FnOnce(T) -> R, R>(self, f: F) -> R;
}

impl<T> Then<T> for T {
    fn then<F: FnOnce(T) -> R, R>(self, f: F) -> R {
        f(self)
    }
}

trait Tap: Sized {
    fn tap<F: FnOnce(&mut Self)>(self, f: F) -> Self;
}

impl<T> Tap for T {
    fn tap<F: FnOnce(&mut Self)>(mut self, f: F) -> Self {
        f(&mut self);
        self
    }
}

fn _day5() {
    let inputs = [
        "47|53\n\
         97|13\n\
         97|61\n\
         97|47\n\
         75|29\n\
         61|13\n\
         75|53\n\
         29|13\n\
         97|29\n\
         53|29\n\
         61|53\n\
         97|53\n\
         61|29\n\
         47|13\n\
         75|47\n\
         97|75\n\
         47|61\n\
         75|61\n\
         47|29\n\
         75|13\n\
         53|13\n\
         \n\
         75,47,61,53,29\n\
         97,61,53,29,13\n\
         75,29,13\n\
         75,97,47,61,53\n\
         61,13,29\n\
         97,13,75,29,47\n"
            .to_string(),
        read_to_string("input05").unwrap(),
    ];
    let input = inputs[1].clone();
    let (ordering_raw, page_num_updates_raw) = input.split_once("\n\n").unwrap();

    fn parse_ordering(raw: &str) -> (i32, i32) {
        raw.split("|")
            .map(|s| s.parse().unwrap())
            .collect_tuple()
            .unwrap()
    }
    let page_orderings: HashSet<(_, _)> = ordering_raw.lines().map(parse_ordering).collect();
    let updates: Vec<Vec<_>> = page_num_updates_raw
        .lines()
        .map(|row| row.split(",").map(|s| s.parse().unwrap()).collect())
        .collect();

    let is_ordered = |pages: &&Vec<_>| {
        let mut pairs = pages.iter().zip(pages.iter().skip(1));
        pairs.all(|(a, b)| page_orderings.contains(&(*a, *b)))
    };
    let (ordered, unordered): (Vec<_>, Vec<_>) = updates.iter().partition(is_ordered);

    fn middle<T: AsRef<[i32]>>(slice: T) -> i32 {
        slice.as_ref().then(|s| s[s.len() / 2])
    }
    println!("{:#?}", ordered.iter().map(middle).sum::<i32>());

    let reordered_middle = |pages: &&Vec<_>| {
        let mut new_pages = pages.to_vec();
        new_pages.sort_by(|p1, p2| match page_orderings.contains(&(*p1, *p2)) {
            true => Ordering::Less,
            false => Ordering::Greater,
        });
        middle(new_pages)
    };
    println!("{:#?}", unordered.iter().map(reordered_middle).sum::<i32>());
}

fn _day4() {
    let inputs = [
        "XMAS\n".to_string(),
        "..X...\n\
         .SAMX.\n\
         .A..A.\n\
         XMAS.S\n\
         .X....\n"
            .to_string(),
        "MMMSXXMASM\n\
         MSAMXMSMSA\n\
         AMXSXMAAMM\n\
         MSAMASMSMX\n\
         XMASAMXAMM\n\
         XXAMMXXAMA\n\
         SMSMSASXSS\n\
         SAXAMASAAA\n\
         MAMMMXMMMM\n\
         MXMXAXMASX\n"
            .to_string(),
        "M.S\n\
         .A.\n\
         M.S\n"
            .to_string(),
        ".M.S......\n\
         ..A..MSMS.\n\
         .M.S.MAA..\n\
         ..A.ASMSM.\n\
         .M.S.M....\n\
         ..........\n\
         S.S.S.S.S.\n\
         .A.A.A.A..\n\
         M.M.M.M.M.\n\
         ..........\n"
            .to_string(),
        read_to_string("input04").unwrap(),
    ];

    let input = inputs[5].clone();

    fn scan(grid: &Vec<Vec<char>>, word: &str, j: i32, i: i32, dir: (i32, i32)) -> i32 {
        let bounds = (
            0..grid.len() as i32,
            0..grid.first().map_or(0, |row| row.len() as i32),
        );

        if !bounds.0.contains(&j) || !bounds.1.contains(&i) {
            return 0;
        }

        let c = grid[j as usize][i as usize];

        if !word.starts_with(&c.to_string()) {
            return 0;
        } else if word.len() == 1 {
            return 1;
        }

        // println!("{:#?}", (word, j, i, dj, di));

        let (dj, di) = dir;
        scan(grid, &word[1..], j + dj, i + di, dir)
    }

    let chars: Vec<Vec<char>> = input.lines().map(|line| line.chars().collect()).collect();
    let height = chars.len() as i32;
    let width = chars.first().map_or(0, |row| row.len() as i32);
    let num_xmas: i32 = iproduct!(0..height, 0..width)
        .map(|(j, i)| {
            iproduct!(-1..=1, -1..=1)
                .map(|dir| scan(&chars, "XMAS", j, i, dir))
                .sum::<i32>()
        })
        .sum();

    // println!("{:#?}", grid);
    println!("{:#?}", num_xmas);

    let deltas = [(-1, -1), (-1, 1), (1, 1), (1, -1)];
    let num_x_mas: i32 = iproduct!(1..height - 1, 1..width - 1)
        .filter(|(j, i)| chars[*j as usize][*i as usize] == 'A')
        .flat_map(|(j, i)| {
            std::iter::repeat(deltas.map(|(dj, di)| (j + dj, i + di)))
                .enumerate()
                .map(|(i, coords)| coords.tap(|c| c.rotate_right(i)))
                .take(4)
        })
        .filter(|rot_coords| {
            rot_coords
                .iter()
                .zip("MMSS".chars())
                .all(|((j, i), c)| chars[*j as usize][*i as usize] == c)
        })
        .count() as i32;
    println!("{:#?}", num_x_mas);
}

fn _day3() {
    let _test_input =
        "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))".to_string();
    let _raw_input = read_to_string("input03").unwrap(); // panic on possible file-reading errors

    fn do_mul_sum(input: &str) -> i32 {
        let re = regex::Regex::new(r"mul\((\d+),(\d+)\)").unwrap();
        re.captures_iter(input)
            .map(|c| {
                let a = c.get(1).unwrap().as_str().parse::<i32>().unwrap();
                let b = c.get(2).unwrap().as_str().parse::<i32>().unwrap();
                // println!("{:#?}", a * b);
                a * b
            })
            .sum()
    }

    println!("{:#?}", do_mul_sum(&_raw_input));

    let _test2_input = "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))";
    let _test3_input =
        "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()do()?mul(8,5))";

    let do_without_donts = _raw_input
        .split("do()")
        .map(|s| s.split_once("don't()").map_or(s, |s| s.0));
    println!("{:#?}", do_without_donts.map(do_mul_sum).sum::<i32>());
}

fn _day2() {
    let _test_input = "\
        7 6 4 2 1\n\
        1 2 7 8 9\n\
        9 7 6 2 1\n\
        1 3 2 4 5\n\
        8 6 4 4 1\n\
        1 3 6 7 9\n"
        .to_string();

    let _raw_input = read_to_string("input02").unwrap(); // panic on possible file-reading errors

    fn parse(input: String) -> Vec<Vec<i32>> {
        input
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|s| s.parse::<i32>().unwrap())
                    .collect()
            })
            .collect()
    }

    fn is_safe(report: &[i32]) -> bool {
        let pairs = report.iter().zip(report.iter().skip(1));
        let diffs: Vec<i32> = pairs.map(|(a, b)| b - a).collect();

        diffs.iter().all(|&d| 0 < d && d < 4) || diffs.iter().all(|&d| -4 < d && d < 0)
    }

    let lines = parse(_raw_input);
    let num_safe = lines.iter().filter(|l| is_safe(l)).count();
    println!("{:#?}", num_safe);

    fn is_dampened_safe(report: &[i32]) -> bool {
        is_safe(report)
            || (0..report.len()).any(|i| {
                let dampened: Vec<i32> = [&report[..i], &report[i + 1..]].concat();
                is_safe(&dampened)
            })
    }

    let num_dampened_safe = lines.iter().filter(|l| is_dampened_safe(l)).count();
    println!("{:#?}", num_dampened_safe);
}

fn _day1() {
    let _test_input = "\
        3   4\n\
        4   3\n\
        2   5\n\
        1   3\n\
        3   9\n\
        3   3\n"
        .to_string();

    let _raw_input = read_to_string("input01").unwrap(); // panic on possible file-reading errors

    fn to_pairs(line: &str) -> (i32, i32) {
        line.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect_tuple()
            .unwrap()
    }
    let lines: Vec<_> = _raw_input.lines().map(to_pairs).collect();

    // let head: Vec<_> = lines.iter().take(10).copied().collect();
    // let _sol = head.into_iter().map(|(a, b)| format!("{a} {b}")).join("\n");
    // println!("{_sol}");

    let (v1, v2): (Vec<_>, Vec<_>) = lines.into_iter().unzip();
    // println!("{:?}", v1.into_iter().sorted().collect::<Vec<_>>());
    // println!("{:?}", v2.into_iter().sorted().collect::<Vec<_>>());

    let s1 = v1.iter().sorted().collect::<Vec<_>>();
    let s2 = v2.iter().sorted().collect::<Vec<_>>();

    let pairs: Vec<_> = s1.into_iter().zip(s2).collect();
    // println!("{:#?}", pairs);
    let sum_diffs: i32 = pairs.iter().map(|(a, b)| (*a - *b).abs()).sum();
    println!("{:#?}", sum_diffs);

    let counts = v2.into_iter().counts();
    // println!("{:#?}", counts);

    // let scores = v1.clone().into_iter()
    //     .map(|v| *counts.get(&v).unwrap_or(&0) as i32)
    //     .collect::<Vec<_>>();
    // println!("{:#?}", scores);

    let sim_score = v1
        .into_iter()
        .map(|v| v * *counts.get(&v).unwrap_or(&0) as i32)
        .sum::<i32>();
    println!("{:#?}", sim_score);
}

// fn input_raw(day: u32) -> Result<String, Box<dyn Error>> {
//     let url = format!("https://adventofcode.com/2024/day/{day}/input");
//     let resp = reqwest::blocking::get(&url)?;
//     let body = resp.text()?;
//     Ok(body)
// }

// fn httpbin() -> Result<String, Box<dyn Error>> {
//     let resp = reqwest::blocking::get("https://httpbin.org/ip")?;
//     let body = resp.text()?;
//     Ok(body)
// }
