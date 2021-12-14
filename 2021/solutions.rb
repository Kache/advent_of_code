#!/usr/bin/env ruby

require 'bundler/inline'

gemfile do
  source 'https://rubygems.org'
  ruby '~> 3.0.2'
  gem 'pry-byebug'
  gem 'activesupport'
end

require 'net/http'
require 'matrix'
require 'active_support/core_ext/enumerable'

module Input
  def self.rows(day, sep = $/)
    raw(day).lines(sep, chomp: true)
  end

  def self.ints(day)
    rows(day).map { _1.split(',').map(&:to_i) }
  end

  def self.raw(day, year = 2021)
    fname = "input#{day.to_s.rjust(2, '0')}"
    return File.read(fname) if File.exists?(fname)

    uri = URI("https://adventofcode.com/#{year}/day/#{day}/input")
    headers = { Cookie: 'session=' + ENV.fetch('AOC_SESSION')}
    Net::HTTP.get(uri, headers).tap { File.write(fname, _1) }
  end
end

module Day14
  def self.occurrence_difference_10
    polymer, rules = input

    10.times do
      inserted = polymer.chars.each_cons(2).map { rules[_1 + _2] }
      polymer = polymer.chars.zip(inserted).flatten.compact.join
    end

    min, max = polymer.chars.tally.minmax_by(&:last).map(&:last)
    max - min
  end

  def self.occurrence_difference_40
    polymer, rules = input
    tally = Hash.new(0).merge(polymer.chars.tally)
    bigrams = polymer.chars.each_cons(2).map(&:join).tally

    40.times do
      bigrams = bigrams.each.with_object(Hash.new(0)) do |(bg, n), new_bigrams|
        new_bigrams[bg[0] + rules[bg]] += n
        new_bigrams[rules[bg] + bg[1]] += n
        tally[rules[bg]] += n
      end
    end

    min, max = tally.minmax_by(&:last).map(&:last)
    max - min
  end

  def self.input
    template, insert_rules = Input.raw(14).split("\n\n")
    [template, insert_rules.lines(chomp: true).map { _1.split(' -> ') }.to_h]
  end
end

module Day13
  def self.num_dots
    dots, folds = input
    fold(dots, *folds.first).size
  end

  def self.display_code
    dots, folds = input
    folds.each { |fold| dots = fold(dots, *fold) }
    display(dots)
  end

  def self.input
    ints, folds = Input.raw(13).split("\n\n")
    dots = ints.split("\n").map { Vector[*_1.split(',').map(&:to_i)] }
    folds = folds.scan(/fold along (.)=(\d+)/).map { |d, i| [d.to_sym, i.to_i] }
    return dots, folds
  end

  def self.fold(dots, dim, i)
    d = { x: 0, y: 1 }[dim]
    lesser, greater = dots.partition { |vec| vec[d] < i }
    flipped = greater.map(&:dup).each { |vec| vec[d] = 2 * i - vec[d] }
    (lesser + flipped).uniq
  end

  def self.display(dots)
    dots = dots.to_set
    x_max, y_max = dots.map { _1[0] }.max, dots.map { _1[1] }.max

    puts '---'
    (0..y_max).each do |y|
      (0..x_max).each do |x|
        print dots.include?(Vector[x, y]) ? '#' : '.'
      end
      puts
    end
  end
end

module Day12
  def self.num_paths
    dfs_paths(start: 'start') do |n, path|
      n == n.upcase || !path.include?(n) # big cave or unvisited
    end.size
  end

  def self.num_paths_revisit_once
    dfs_paths(start: 'start') do |n, path|
      n == n.upcase || !path.include?(n) ||
        (n != 'start' && n != 'end' && path.tally.all? { _1 == _1.upcase || _2 < 2 })
    end.size
  end

  def self.dfs_paths(start: 'start')
    stack = [['start', 0]]
    path, paths = [], []

    until stack.empty?
      curr, lvl = stack.pop

      path.pop until path.size == lvl
      path << curr

      if curr == 'end'
        paths << path.dup
      else
        neighbors = graph[curr].select { yield _1, path }
        neighbors.sort.reverse_each { stack << [_1, lvl + 1] }
      end
    end

    paths
  end

  def self.graph
    @graph ||= input.each_with_object({}) do |(a, b), graph|
      (graph[a] ||= Set.new) << b
      (graph[b] ||= Set.new) << a
    end
  end

  def self.input
    Input.rows(12).map { _1.split('-') }
  end
end

module Day11
  def self.num_flashes
    octopi, flashes = input, 0

    100.times do
      octopi = step(octopi)
      flashes += octopi.flatten.count { _1 == FLASHED }
    end

    flashes
  end

  def self.first_sync
    octopi = input

    (1..).find do |n|
      octopi = step(octopi)
      octopi.all? { _1.all?(&:zero?) }
    end
  end

  def self.input
    Input.ints(11).flatten.map { _1.digits.reverse }
  end

  FLASH, FLASHED = 10, 0

  def self.step(octopi)
    octopi = grid(octopi) { _1 + 1 }

    while octopi.any? { |row| row.any? { _1 == FLASH } }
      octopi = grid(octopi) do |o, neighbors|
        next o if o == FLASHED
        next FLASHED if o == FLASH
        (o + neighbors.count { _1 == FLASH }).clamp(0, 10)
      end
    end

    octopi
  end

  def self.grid(octopi)
    (0..9).map do |j|
      (0..9).map do |i|
        coord = [j, i]
        coord_box = (-1..1).map { _1 + j }.product((-1..1).map { _1 + i })
        adj_coords = coord_box.select { |p| p.all? { (0..9).cover?(_1) } && p != coord }

        yield octopi[j][i], adj_coords.map { octopi[_1][_2] }
      end
    end
  end

  def self.display(octopi)
    puts '---'
    octopi.each do |row|
      puts row.map { _1.zero? ? '█' : _1 }.join
    end
  end
end

module Day10
  def self.syntax_error_score
    scoring = { ')' => 3, ']' => 57, '}' => 1197, '>' => 25137 }
    Input.rows(10).map { parse(_1)[:corrupted] }.compact.sum(&scoring)
  end

  def self.middle_score
    stacks = Input.rows(10).map { parse(_1)[:incomplete] }.compact
    completions = stacks.map { _1.reverse.map(&CLOSES) }

    scoring = { ')' => 1, ']' => 2, '}' => 3, '>' => 4 }
    scores = completions.map { _1.reduce(0) { |score, c| score * 5 + scoring[c] } }
    scores.sort[scores.length / 2]
  end

  CLOSES = { '(' => ')', '[' => ']', '{' => '}', '<' => '>' }

  def self.parse(line)
    stack = []

    line.chars.each do |c|
      if c == CLOSES[stack.last]
        stack.pop
      elsif CLOSES.key?(c)
        stack << c
      else
        return { corrupted: c }
      end
    end

    return { incomplete: stack }
  end
end

module Day09
  def self.low_risk_sum
    graph.select { _1 == _2 }.keys.sum { height(_1) + 1 }
  end

  def self.basin_size_mult
    flow_from = graph.each_with_object({}) do |(from, to), graph|
      (graph[to] ||= Set.new) << from
    end

    basins = graph.select { _1 == _2 }.transform_values do |start|
      basin = Set.new
      queue = [start]
      until queue.empty?
        curr = queue.shift
        basin << curr
        (flow_from.fetch(curr, Set.new) - basin).inject(queue, :<<)
      end
      basin
    end

    basins.transform_values(&:size).values.sort.last(3).inject(:*)
  end

  def self.graph
    @graph ||= grid.each.with_object({}) do |(pt, adj), graph|
      graph[pt] = [pt, *adj].select { height(_1) < 9 }.min_by { height(_1) } if height(pt) < 9
    end
  end

  def self.map
    @map ||= Input.rows(9).map { _1.split('').map(&:to_i) }
  end

  def self.height(coord)
    map[coord[0]][coord[1]]
  end

  def self.grid
    return enum_for(:grid) unless block_given?

    j_idxs, i_idxs = (0...map.size), (0...map.first.size)
    j_idxs.each do |j|
      i_idxs.each do |i|
        adj = [[j + 1, i], [j - 1, i], [j, i + 1], [j, i - 1]]
        yield [j, i], adj.select { j_idxs.cover?(_1) && i_idxs.cover?(_2) }
      end
    end
  end
end

module Day08
  def self.num_appearances
    by_size = SEGMENTS.slice(1, 4, 7, 8).transform_values(&:size).invert
    input.map(&:last).map { |sigs| sigs.count { by_size.key?(_1.size) } }.sum
  end

  def self.output_sum
    input.map do |patterns, output|
      by_len = patterns.group_by(&:size)
      pat = [1, 7, 4, 8].zip(by_len.values_at(2, 3, 4, 7).flatten).to_h
      pat[6] = by_len[6].find { _1 + pat[1] == pat[8] }
      pat[2] = by_len[5].find { _1 + pat[4] == pat[8] }
      pat[5] = by_len[5].find { _1 < pat[6] }
      pat[0] = by_len[6].find { pat[4] - pat[1] + _1 == pat[8] }

      map_back = {
        a: pat[7] - pat[1],
        b: pat[8] - pat[2] - pat[1],
        c: pat[8] - pat[6],
        d: pat[8] - pat[0],
        e: pat[6] - pat[5],
        f: pat[7] - pat[2],
        g: pat[5] - pat[7] - pat[4],
      }.transform_values(&:first).invert

      output.map { _1.to_set(&map_back) }.map(&SEGMENTS.invert).join.to_i
    end.sum
  end

  SEGMENTS = {
    0 => %i[a b c e f g],
    1 => %i[c f],
    2 => %i[a c d e g],
    3 => %i[a c d f g],
    4 => %i[b c d f],
    5 => %i[a b d f g],
    6 => %i[a b d e f g],
    7 => %i[a c f],
    8 => %i[a b c d e f g],
    9 => %i[a b c d f g],
  }.transform_values(&:to_set)

  def self.input
    Input.rows(8).map do |line|
      line.split(' | ').map do |part|
        part.split(' ').map { _1.chars.map(&:to_sym).to_set }
      end
    end
  end
end

module Day07
  def self.fuel_amount_linear
    fuel_amount { _1 }
  end

  def self.fuel_amount_geometric
    fuel_amount { |n| n * (n + 1) / 2 }
  end

  def self.fuel_amount
    crabs = Input.ints(7).first.tally

    (crabs.keys.min..crabs.keys.max).map do |target|
      crabs.sum { |pos, num| num * (yield (target - pos).abs) }
    end.min
  end
end

module Day06
  def self.num_fish_80
    num_fish(80)
  end

  def self.num_fish_256
    num_fish(256)
  end

  def self.num_fish(n)
    fish_cycle = (0..8).index_with { 0 }.merge(Input.ints(6).first.tally)
    n.times do
      num_born = fish_cycle[0]
      (0..8).each_cons(2) { fish_cycle[_1] = fish_cycle[_2] }
      fish_cycle[8] = num_born
      fish_cycle[6] += num_born
    end
    fish_cycle.values.sum
  end
end

module Day05
  def self.num_overlapping_cardinal
    cardinal_lines = input.select { |h, t| h[0] == t[0] || h[1] == t[1] }
    cardinal_lines.flat_map { |head, tail| interp_coords(head, tail) }.tally.count { _2 > 1 }
  end

  def self.num_overlapping
    input.flat_map { |head, tail| interp_coords(head, tail) }.tally.count { _2 > 1 }
  end

  def self.input
    Input.rows(5).map do |line|
      head, tail = line.split(' -> ').map { Vector[*_1.split(',').map(&:to_i)] }
    end
  end

  def self.interp_coords(v1, v2)
    dimensions = Matrix[v1, v2].column_vectors.map(&:to_a)

    dim_vals = dimensions.map do |a, b|
      change = b <=> a
      change.zero? ? [a] : (a..b).step(change).to_a
    end
    max_size = dim_vals.map(&:size).max
    x_vals, y_vals = dim_vals.map { _1 * (max_size / _1.size) }

    x_vals.zip(y_vals)
  end
end

module Day04
  def self.winning_board_score
    board_placements.first.score
  end

  def self.losing_board_score
    board_placements.last.score
  end

  def self.board_placements
    nums, boards = input
    nums.flat_map do |n|
      boards.select { _1.score.zero? }.each { _1.mark!(n) }.select { _1.score.positive? }
    end
  end

  def self.input
    nums, *boards = Input.rows(4, "\n\n")
    bingo_nums = nums.split(',').map(&:to_i)
    bingo_boards = boards.map { |b| b.lines(chomp: true).map(&:split).map { _1.map(&:to_i) } }
    return bingo_nums, bingo_boards.map { Board.new(_1) }
  end

  class Board
    attr_reader :score

    def initialize(board)
      @nums = board.flatten.to_set
      @sets = [@nums] + board.map(&:to_set) + board.transpose.map(&:to_set)
      @score = 0
    end

    def mark!(num)
      @sets.each { _1.delete(num) } if @score.zero?
      @score = @sets.any?(&:empty?) ? @nums.sum * num : 0
      self
    end
  end
end

module Day03
  def self.power_consumption
    freq_bits = frequent_bitvals(input)
    to_num(freq_bits) * to_num(freq_bits.map { 1 - _1 })
  end

  def self.life_support_rating
    oxy = successive_bitwise_filter(input) { frequent_bitvals(_1) }
    co2 = successive_bitwise_filter(input) { frequent_bitvals(_1).map { |b| 1 - b } }
    to_num(oxy) * to_num(co2)
  end

  def self.input
    Input.rows(3).map { _1.split('').map(&:to_i) }
  end

  def self.frequent_bitvals(bits_arr)
    vec_sum = bits_arr.map { Vector[*_1] }.inject(:+)
    vec_sum.to_a.map { _1 * 2 < bits_arr.length ? 0 : 1 }
  end

  def self.to_num(bits)
    bits.map(&:to_s).join.to_i(2)
  end

  def self.successive_bitwise_filter(candidates)
    (0...candidates.first.size).each do |i|
      filter_bits = yield candidates
      candidates = candidates.select { _1[i] == filter_bits[i] }
      break candidates.first if candidates.one?
    end
  end
end

module Day02
  def self.coordinate_loc_mult
    loc = input.map { |dir, dist| @vec[dir] * dist }.inject(:+)
    loc.map(&:abs).inject(:*)
  end

  def self.aim_loc_mult
    aim = zero = Vector.zero(2)
    loc = input.map do |dir, dist|
      aim += Vector[0, @vec[dir][1] * dist]

      dir == :forward ? @vec[dir] * dist + aim * dist : zero
    end.inject(:+)
    loc.map(&:abs).inject(:*)
  end

  @vec = {
    forward: Vector[1, 0],
    down:    Vector[0, -1],
    up:      Vector[0, 1],
  }

  @aim = { down: 1, up: -1 }

  def self.input
    Input.rows(2).map do
      dir, dist = _1.split(' ')
      [dir.to_sym, dist.to_i]
    end
  end
end

module Day01
  def self.num_larger
    Input.rows(1).map(&:to_i).each_cons(2).count { _2 > _1 }
  end

  def self.num_sums_larger
    sums = Input.rows(1).map(&:to_i).each_cons(3).map(&:sum)
    sums.each_cons(2).count { _2 > _1 }
  end
end
