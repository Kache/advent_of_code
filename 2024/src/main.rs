// This is a comment, and is ignored by the compiler.
// You can test this code by clicking the "Run" button over there ->
// or if you prefer to use your keyboard, you can use the "Ctrl + Enter"
// shortcut.

// This code is editable, feel free to hack it!
// You can always return to the original code by clicking the "Reset" button ->

// use std::error::Error;
// use std::{clone, fs::read_to_string};
use itertools::Itertools;
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

    // _day1();

    // _day2();

    _day3();
}

fn _day3() {
    let _test_input =
        "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))".to_string();
    let _raw_input = read_to_string("input03").unwrap(); // panic on possible file-reading errors

    fn do_mul_sum(input: &String) -> i32 {
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

    let _test2_input =
        "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))".to_string();
    let _test3_input =
        "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()do()?mul(8,5))".to_string();

    let do_dont_sum: i32 = _raw_input
        .split("do()")
        .map(|s| do_mul_sum(&s.split("don't()").next().unwrap().to_string()))
        .sum();
    println!("{:#?}", do_dont_sum);
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

    fn is_safe(report: &Vec<i32>) -> bool {
        let diffs: Vec<i32> = report
            .iter()
            .zip(report.iter().skip(1))
            .map(|(a, b)| b - a)
            .collect();

        diffs.iter().all(|&d| 0 < d && d < 4) || diffs.iter().all(|&d| -4 < d && d < 0)
    }

    let lines = parse(_raw_input);
    let num_safe = lines.iter().filter(|l| is_safe(l)).count();
    println!("{:#?}", num_safe);

    fn is_dampened_safe(report: &Vec<i32>) -> bool {
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

    fn parse(input: String) -> Vec<(i32, i32)> {
        input
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|s| s.parse::<i32>().unwrap())
                    .collect_tuple::<(i32, i32)>()
                    .unwrap()
            })
            .collect()
    }

    let lines = parse(_raw_input);

    let head: Vec<_> = lines.iter().take(10).map(|l| l.clone()).collect();
    let _sol = head.into_iter().map(|(a, b)| format!("{a} {b}")).join("\n");

    // println!("{_sol}");

    let (v1, v2): (Vec<_>, Vec<_>) = lines.into_iter().unzip();
    // println!("{:?}", v1.into_iter().sorted().collect::<Vec<_>>());
    // println!("{:?}", v2.into_iter().sorted().collect::<Vec<_>>());

    let s1 = v1.clone().into_iter().sorted().collect::<Vec<_>>();
    let s2 = v2.clone().into_iter().sorted().collect::<Vec<_>>();

    let pairs: Vec<_> = s1.into_iter().zip(s2.into_iter()).collect();
    // println!("{:#?}", pairs);
    let sum_diffs: i32 = pairs.iter().map(|(a, b)| (a - b).abs()).sum();
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
