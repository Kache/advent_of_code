#!/usr/bin/env ruby

require 'bundler/inline'

gemfile do
  source 'https://rubygems.org'
  ruby '~> 3.0.2'
  gem 'pry-byebug'
  gem 'activesupport', require: 'active_support/core_ext/enumerable'
  gem 'priority_queue_cxx', require: 'fc'
end

require 'net/http'
require 'matrix'

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

module Day19
  def self.num_beacons
    oriented_scanners.values.inject(&:union).size
  end

  def self.manhattan_diameter
    oriented_scanners.keys.map(&:pos).combination(2).map { |a, b| (a - b).sum(&:abs) }.max
  end

  ROT = %i[x y z].index_with.with_index do |_, i|
    signs = [1, -1, 1].rotate(-i)
    rows = [Vector[1, 0, 0], Vector[0, 0, 1], Vector[0, 1, 0]].rotate(i)
    Matrix[*signs.zip(rows).map { _1 * _2 }]
  end

  ROT24 = {
    ROT[:z]**0 => :x, ROT[:z]**1 => :y, ROT[:y]**1 => :z,
    ROT[:z]**2 => :x, ROT[:z]**3 => :y, ROT[:y]**3 => :z,
  }.flat_map { |heading, roll| (0..3).map { heading * ROT[roll]**_1 } }

  FrameOfRef = Struct.new('FrameOfRef', :rot, :pos) do
    def self.frame_for(rot, refs, vecs)
      matches = refs[0] - refs[1] == rot * (vecs[0] - vecs[1])
      # i.e. FrameOfRef.new(rot, refs[0] - rot * vecs[0]) == FrameOfRef.new(rot, refs[1] - rot * vecs[1])
      # i.e. FrameOfRef.new(rot, refs[0] - rot * vecs[0]).interpret(vecs[1]) == refs[1]
      FrameOfRef.new(rot, refs[0] - rot * vecs[0]) if matches
    end

    def interpret(relative_vector)
      rot * relative_vector + pos
    end

    def normalize(rel_frame)
      FrameOfRef.new(rot * rel_frame.rot, interpret(rel_frame.pos))
    end
  end

  def self.oriented_scanners
    @oriented_scanners ||= begin
      coord_dists = scanners.transform_values do |coords|
        coords.combination(2).group_by { |a, b| (a - b).sum { _1**2 } }
      end

      pair_matches = coord_dists.keys.permutation(2).index_with do |a, b|
        coord_dists[a].flat_map do |dist, a_pairs|
          b_pairs = coord_dists[b].fetch(dist, [])
          a_pairs.product(b_pairs)
        end
      end.sort_by { -_2.size }.to_h

      ref_frames = { 0 => FrameOfRef.new(Matrix.identity(3), Vector.zero(3)) }

      while (pairing = pair_matches.detect { |(a, b), _| ref_frames.key?(a) && !ref_frames.key?(b) })
        (ref, rel), matching_coord_pairs = pairing

        possible_frames = matching_coord_pairs.map do |refs, vecs|
          ROT24.lazy.map do |rot|
            FrameOfRef.frame_for(rot, refs, vecs) || FrameOfRef.frame_for(rot, refs, vecs.reverse)
          end.detect(&:itself)
        end.compact

        matching_rel_frame, _ = possible_frames.tally.sort_by(&:last).last
        ref_frames[rel] = ref_frames[ref].normalize(matching_rel_frame)
      end

      scanners.each_with_object({}) do |(i, coords), oriented|
        oriented[ref_frames[i]] = coords.to_set { ref_frames[i].interpret(_1) }
      end
    end
  end

  def self.scanners
    @scanners ||= Input.raw(19).split("\n\n").map { _1.split("\n") }.map do |name, *coords|
      coords.map { Vector[*_1.split(',').map(&:to_i)] }
    end.each.with_index.to_h.invert
  end
end

module Day18
  def self.mag_final_sum
    mag(snums.inject { add_snum(_1, _2) })
  end

  def self.two_sum_mag_max
    snums.permutation(2).map { mag(add_snum(*_1)) }.max
  end

  def self.snums
    Input.raw(18).each_char.with_object([[]]) do |c, stack|
      case c
      when '['      then stack << []
      when '0'..'9' then stack.last << c.to_i
      when ']'      then stack[-2] << stack.pop
      end
    end.pop
  end

  def self.add_snum(a, b)
    exploded, _ = explode([a, b])
    return add_snum(*exploded) if [a, b] != exploded
    splitted = split(exploded)
    return add_snum(*splitted) if [a, b] != splitted
    splitted
  end

  LEFT = 0; RIGHT = 1

  def self.explode(snum, depth = 0, dir = nil)
    return 0, snum if depth == 4
    return explode(snum, depth, LEFT) || explode(snum, depth, RIGHT) || [snum, nil] if dir.nil?

    a, b = snum.rotate(dir)
    new_a, explosion = explode(a, depth + 1) if a.is_a?(Array)
    exp_a, exp_b = explosion.rotate(dir) if explosion
    new_b = exp_b ? explode_into(b, exp_b, dir) : b
    return [new_a, new_b].rotate(dir), [exp_a, nil].rotate(dir) if new_a && new_a != a
  end

  def self.explode_into(snum, exp_val, dir)
    return snum + exp_val unless snum.is_a?(Array)
    elem_hit, other = snum.rotate(dir)
    [explode_into(elem_hit, exp_val, dir), other].rotate(dir)
  end

  def self.split(snum, dir = nil)
    return split(snum, LEFT) || split(snum, RIGHT) || snum if dir.nil?

    s, other = snum.rotate(dir)
    split = s.is_a?(Array) ? split(s) : s > 9 && [s / 2, (s + 1) / 2]
    return [split, other].rotate(dir) if split && !split.equal?(s)
  end

  def self.mag(snum)
    snum.is_a?(Array) ? 3 * mag(snum[0]) + 2 * mag(snum[1]) : snum
  end
end

module Day17
  def self.max_y
    vel = all_velocities.max_by { |v| v[1] }
    launch(vel, display: true).map { |v| v[1] }.max
  end

  def self.num_velocities
    all_velocities(display: true).size
  end

  def self.all_velocities(display: false)
    queue, solutions, attempted = target.dup, Set.new, Set.new

    until queue.empty?
      curr = queue.first.tap { queue.delete(_1) }
      _start, vel, *flight = path = launch(curr)

      attempted << vel
      next unless target.member?(path.last)
      solutions << vel

      nearby_vels = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]].map { vel + Vector[_1, _2] }
      lob_higher = vel * (path.size - 1) / path.size.to_f + Vector[1, flight.size]
      [*nearby_vels, lob_higher.map(&:to_i)].reject { attempted.member?(_1) }.inject(queue, :<<)
    end

    display_searchspace(attempted, solutions) if display
    solutions
  end

  def self.launch(vel, display: false)
    _, x_max, y_min, _ = input
    path = [Vector[0, 0]]

    until target.member?(path.last) || path.last[0] > x_max || path.last[1] < y_min
      path << path.last.dup + vel
      vel -= Vector[vel[0] <=> 0, 1]
    end

    display(path) if display
    path
  end

  def self.display(path)
    path = path.to_set
    print_grid(path | target, max_y: 3) do |pt|
      pt == path.first     ? 'S' :
        path.member?(pt)   ? '#' :
        target.member?(pt) ? 'T' :
                             '.'
    end
  end

  def self.display_searchspace(attempted, solutions)
    print_grid(attempted) do |pt|
      solutions.member?(pt)   ? '#' :
        attempted.member?(pt) ? '.' :
                                ' '
    end
    puts "attempted: #{attempted.size}"
  end

  def self.print_grid(points, max_y: Float::INFINITY)
    y_min, y_max = points.map { _1[1] }.minmax
    x_min, x_max = points.map { _1[0] }.minmax

    ([y_max, max_y].min..y_min).step(-1).each do |y|
      (x_min..x_max).each { |x| print yield Vector[x, y] }
      puts
    end
  end

  def self.target
    @target ||= begin
      x0, x1, y0, y1 = input
      (x0..x1).to_a.product((y0..y1).to_a).to_set { Vector[_1, _2] }
    end
  end

  def self.input
    @input ||= /target area: x=(-?\d+)..(-?\d+), y=(-?\d+)..(-?\d+)/.match(Input.raw(17)).captures.map(&:to_i)
  end
end

module Day16
  def self.version_nums_sum(packet = Packet.from(input))
    packet.type == :lit ? packet.ver : packet.ver + packet.val.sum { version_nums_sum(_1) }
  end

  def self.evaluate
    Packet.from(input).eval
  end

  def self.input
    Input.rows(16).first.prepend('1').to_i(16).to_s(2)[1..]
  end

  refine String do
    define_method(:take!) { slice!(0, _1) }
  end

  Packet = Struct.new(:ver, :type, :val) do
    TYPE = { sum: 0, mul: 1, min: 2, max: 3, lit: 4, gt: 5, lt: 6, eq: 7 }.invert
    using Day16

    def self.from(bits)
      pack = new(bits.take!(3).to_i(2), TYPE[bits.take!(3).to_i(2)])

      if pack.type == :lit
        (chunks ||= []) << bits.take!(5) until chunks&.last&.start_with?('0')
        pack.val = chunks.map { _1[1..] }.join.to_i(2)
      elsif bits.take!(1).to_i.zero?
        sub_bits = bits.take!(bits.take!(15).to_i(2))
        (pack.val ||= []) << from(sub_bits) until sub_bits.empty?
      else
        pack.val = Array.new(bits.take!(11).to_i(2)) { from(bits) }
      end

      pack
    end

    def eval
      case type
      when :sum then val.sum(&:eval)
      when :mul then val.map(&:eval).inject(:*)
      when :min then val.map(&:eval).min
      when :max then val.map(&:eval).max
      when :lit then val
      when :gt  then val.map(&:eval).inject(:>)  ? 1 : 0
      when :lt  then val.map(&:eval).inject(:<)  ? 1 : 0
      when :eq  then val.map(&:eval).inject(:==) ? 1 : 0
      end
    end
  end
end

module Day15
  def self.min_risk_small
    min_risk(risk_map)
  end

  def self.min_risk_full
    risk_map_5x5 = (0...5).flat_map do |y|
      risk_map.map do |row|
        (0...5).flat_map do |x|
          row.map { |risk| ((risk - 1 + y + x) % 9) + 1 } # +1 to all, rolls 9 over to 1
        end
      end
    end

    min_risk(risk_map_5x5)
  end

  def self.risk_map
    Input.rows(15).map { _1.split('').map(&:to_i) }
  end

  def self.min_risk(map)
    max_j, max_i = map.size - 1, map.first.size - 1

    neighbors = lambda do |(j, i)|
      adj = [[j + 1, i], [j - 1, i], [j, i + 1], [j, i - 1]]
      adj.select { (0..max_j).cover?(_1) && (0..max_i).cover?(_2) }
    end

    dijkstra_dist(neighbors, start: [0, 0], goal: [max_j, max_i]) { |_, (j, i)| map[j][i] }
  end

  def self.dijkstra_dist(neighbors, start:, goal:)
    dists = { start => 0 }
    pqueue = FastContainers::PriorityQueue.new(:min).push(start, 0)

    until (curr = pqueue.pop) == goal
      closer_neighbors = neighbors[curr]
        .index_with { |n| dists[curr] + (yield curr, n) }
        .select { |n, dist| dist < dists.fetch(n, Float::INFINITY) }

      closer_neighbors.each { |n, dist| pqueue.push(n, dists[n] = dist) }
    end

    dists[goal]
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
